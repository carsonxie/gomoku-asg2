"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
import traceback
from sys import stdin, stdout, stderr
from board_util import GoBoardUtil, BLACK, WHITE, EMPTY, BORDER, PASS, \
                       MAXSIZE, coord_to_point
import numpy as np
import re
import signal
import time

class GtpConnection():

    def __init__(self, go_engine, board, debug_mode = False):
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board: 
            Represents the current board state.
        """
        self._debug_mode = debug_mode
        self.go_engine = go_engine
        self.board = board
        self.commands = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "solve":self.solve_cmd, # add a solve mapping in command dictionary
            "timelimit":self.timelimit_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_game_id": self.gogui_rules_game_id_cmd,
            "gogui-rules_board_size": self.gogui_rules_board_size_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_side_to_move": self.gogui_rules_side_to_move_cmd,
            "gogui-rules_board": self.gogui_rules_board_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "gogui-analyze_commands": self.gogui_analyze_cmd
        }

        # used for argument checking
        # values: (required number of arguments, 
        #          error message on argnum failure)
        self.argmap = {
            "boardsize": (1, 'Usage: boardsize INT'),
            "komi": (1, 'Usage: komi FLOAT'),
            "known_command": (1, 'Usage: known_command CMD_NAME'),
            "genmove": (1, 'Usage: genmove {w,b}'),
            "play": (2, 'Usage: play {b,w} MOVE'),
            "legal_moves": (1, 'Usage: legal_moves {w,b}')
        }
    
    def write(self, data):
        stdout.write(data) 

    def flush(self):
        stdout.flush()

    def start_connection(self):
        """
        Start a GTP connection. 
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command):
        """
        Parse command string and execute it
        """
        if len(command.strip(' \r\t')) == 0:
            return
        if command[0] == '#':
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements = command.split()
        if not elements:
            return
        command_name = elements[0]; args = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".
                               format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error('Unknown command')
            stdout.flush()

    def has_arg_error(self, cmd, argnum):
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg):
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg):
        """ Send error msg to stdout """
        stdout.write('? {}\n\n'.format(error_msg))
        stdout.flush()

    def respond(self, response=''):
        """ Send response to stdout """
        stdout.write('= {}\n\n'.format(response))
        stdout.flush()

    def reset(self, size):
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self):
        return str(GoBoardUtil.get_twoD_board(self.board))
        
    def protocol_version_cmd(self, args):
        """ Return the GTP protocol version being used (always 2) """
        self.respond('2')

    def quit_cmd(self, args):
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args):
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args):
        """ Return the version of the  Go engine """
        self.respond(self.go_engine.version)

    def clear_board_cmd(self, args):
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args):
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args):
        self.respond('\n' + self.board2d())

    def komi_cmd(self, args):
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args):
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args):
        """ list all supported GTP commands """
        self.respond(' '.join(list(self.commands.keys())))

    def legal_moves_cmd(self, args):
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        moves = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves = []
        for move in moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)

    def handler(self,sig,frame):
        self.respond("it takes too long!")
        self.board.continue_to_search = False
        return

    def timelimit_cmd(self,args = ["1"]):
        """This command sets the maximum time to 
        use for all following genmove or solve commands, 
        until it is changed by another timelimit command.
        
        The argument seconds is an integer 
        in the range 1 <= seconds <= 100. 
        """
        timelim = args[0]
        try:
            sec = int(timelim)
            if sec < 0 or sec > 100:
                raise Exception
        except Exception:
            self.respond("No propreate arguments for timelimit")
            return

        signal.signal(signal.SIGALRM,self.handler)
        signal.alarm(sec)
        

    def solve_cmd(self,args):
        """
        This function commputes the winner.
        If the winner is current player, or the game is a draw,
        write a move that ahieves this best possible result.
        If the winner is opponent or the result is unknown,
        this function only generate the winner or unknown message
        """


        win_check = self.board.check_game_end_gomoku()
        if win_check[0] == True:
            if args != "flag":
                self.respond(int_to_color(win_check[1]))
            return None

        self.board.start_player = self.board.current_player
        print("self.board.startplayer is "+str(self.board.start_player))
        result = self.board.AlphaBeta(-100,11)
        print("\n\nresult in solve "+str(result))
        print("The win move is "+str(self.board.win_move))
        
        if result == None:
            if args != "flag":
                self.respond("unknow")
            return None

        if result == 1:
            winner = int_to_color(result)
            win_move = self.board.win_move
            if args != "flag":
                self.respond(winner + " " + win_move)
            return (winner,win_move)

        if result == 0:
            if args != "flag":
                self.respond("draw "+self.board.win_move)

        if result == -1:
            if args != "flag":
                self.respond(int_to_color(GoBoardUtil.opponent(self.board.current_player)))



        
    
    #play
    def play_cmd(self, args):
        """
        play a move args[1] for given color args[0] in {'b','w'}
        """
        try:
             # board_color => b / w
             # board_move => A1
            board_color = args[0].lower()
            board_move = args[1] 

            if board_color != "b" and board_color !="w":
                self.respond("illegal move: \"{}\" wrong color".format(board_color))
                return
            
            # 1 => black;  2 => white;
            color = color_to_int(board_color) 

            if args[1].lower() == 'pass':
                self.board.play_move(PASS, color)
                self.board.current_player = GoBoardUtil.opponent(color)
                self.respond()
                return
            
            # tuple (x,y) => coord
            coord = move_to_coord(args[1], self.board.size)
            
            if coord:
                # int x => move
                move = coord_to_point(coord[0],coord[1], self.board.size)

            else:
                self.error("Error executing move {} converted from {}"
                           .format(move, args[1]))
                return
            if not self.board.play_move_gomoku(move, color):
                self.respond("illegal move: \"{}\" occupied".format(board_move))
                return
            else:
                self.debug_msg("Move: {}\nBoard:\n{}\n".
                                format(board_move, self.board2d()))
            self.respond()
        except Exception as e:
            self.respond('{}'.format(str(e)))

    #genmove
    def genmove_cmd(self, args):
        """
        Generate a move for the color args[0] in {'b', 'w'}, for the game of gomoku.
        """
        result = self.solve_cmd("flag")
        print(result)
        if result != None:
            print("result in genmove"+str(result))
            self.respond(result[1])
            return
        
        board_color = args[0].lower()
        color = color_to_int(board_color)
        game_end, winner = self.board.check_game_end_gomoku()
        if game_end:
            if winner == color:
                self.respond("pass")
            else:
                self.respond("resign")
            return
        move = self.go_engine.get_move(self.board, color)
        if move == PASS:
            self.respond("pass")
            return
        move_coord = point_to_coord(move, self.board.size)
        move_as_string = format_point(move_coord)
        if self.board.is_legal_gomoku(move, color):
            self.board.play_move_gomoku(move, color)
            self.respond(move_as_string)
        else:
            self.respond("illegal move: {}".format(move_as_string))

    def gogui_rules_game_id_cmd(self, args):
        self.respond("Gomoku")
    
    def gogui_rules_board_size_cmd(self, args):
        self.respond(str(self.board.size))
    
    def legal_moves_cmd(self, args):
        """
            List legal moves for color args[0] in {'b','w'}
            """
        board_color = args[0].lower()
        color = color_to_int(board_color)
        moves = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves = []
        for move in moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)

    def gogui_rules_legal_moves_cmd(self, args):
        game_end,_ = self.board.check_game_end_gomoku()
        if game_end:
            self.respond()
            return
        moves = GoBoardUtil.generate_legal_moves_gomoku(self.board)
        gtp_moves = []
        for move in moves:
            coords = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = ' '.join(sorted(gtp_moves))
        self.respond(sorted_moves)
    
    def gogui_rules_side_to_move_cmd(self, args):
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)
    
    def gogui_rules_board_cmd(self, args):
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)
    
    def gogui_rules_final_result_cmd(self, args):
        game_end, winner = self.board.check_game_end_gomoku()
        moves = self.board.get_empty_points()
        board_full = (len(moves) == 0)
        if board_full and not game_end:
            self.respond("draw")
            return
        if game_end:
            color = "black" if winner == BLACK else "white"
            self.respond(color)
        else:
            self.respond("unknown")

    def gogui_analyze_cmd(self, args):
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

def point_to_coord(point, boardsize):
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is not transformed
    """
    if point == PASS:
        return PASS
    else:
        NS = boardsize + 1
        return divmod(point, NS)

def format_point(move):
    """
    Return move coordinates as a string such as 'a1', or 'pass'.
    """
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    #column_letters = "abcdefghjklmnopqrstuvwxyz"
    if move == PASS:
        return "pass"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1]+ str(row) 
    
def move_to_coord(point_str, board_size):
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return PASS
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("illegal move: \"{}\" wrong coordinate".format(s))
    if not (col <= board_size and row <= board_size):
        raise ValueError("illegal move: \"{}\" wrong coordinate".format(s))
    return row, col

def color_to_int(c):
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK , "w": WHITE, "e": EMPTY, 
                    "BORDER": BORDER}
    return color_to_int[c] 

def int_to_color(int):
    d = {BLACK:"b",WHITE:"w",EMPTY:"e"}
    return d[int]
