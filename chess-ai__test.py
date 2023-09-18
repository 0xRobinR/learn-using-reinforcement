import chess
import chess.engine
import numpy as np
import tensorflow as tf

engine = chess.engine.SimpleEngine.popen_uci('./utils/stockfish/stockfish.exe')


def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12))

    piece_dict = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                  'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            if piece:
                tensor[i, j, piece_dict[piece.symbol()]] = 1

    return np.expand_dims(tensor, axis=0)


def output_to_move(output, board):
    legal_moves = [move.uci() for move in board.legal_moves]
    predicted_move_index = np.argmax(output)
    predicted_move_uci = legal_moves[predicted_move_index]
    predicted_move = chess.Move.from_uci(predicted_move_uci)

    return predicted_move


model = tf.keras.models.load_model('./models/chess-rl.h5')

model.summary()

for i in range(100):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
            board.push(move)
        else:
            tensor_board = board_to_tensor(board)
            tensor_board = np.expand_dims(tensor_board, axis=0)
            output = model.predict(tensor_board)
            move = output_to_move(output, board)
            board.push(move)

    print("Result #{0}: ".format(i) + board.result())

engine.quit()
