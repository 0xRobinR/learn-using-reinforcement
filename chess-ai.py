import chess.engine
import numpy as np
import tensorflow as tf

engine = chess.engine.SimpleEngine.popen_uci("./utils/stockfish/stockfish.exe")
board = chess.Board()

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8, 8, 12)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4096, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

Q_table = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.3


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


def choose_move(state, board, model):
    legal_moves = [move.uci() for move in board.legal_moves]
    q_values = model.predict(state)[0]

    legal_mask = np.zeros_like(q_values)
    for i, move_uci in enumerate(legal_moves):
        legal_mask[i] = 1
    legal_q_values = q_values * legal_mask
    best_move_index = np.argmax(legal_q_values)
    best_move_uci = legal_moves[best_move_index]

    return best_move_uci, legal_moves


for games in range(10):
    board.reset()
    while not board.is_game_over():
        state = board_to_tensor(board)
        move, legal_moves = choose_move(state, board, model)

        board.push(chess.Move.from_uci(move))

        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        reward = info["score"].relative.score()

        next_state = board_to_tensor(board)
        next_state = np.expand_dims(next_state, axis=0)

        target = reward + gamma * np.max(model.predict(next_state))
        target = tf.expand_dims(target, axis=0)
        with tf.GradientTape() as tape:
            q_values = model(state)
            move_index = legal_moves.index(chess.Move.from_uci(move).uci())
            if move_index != -1:
                print(move_index, q_values[0][move_index])
                loss = tf.keras.losses.MSE(target, q_values[0][move_index])
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"Result: {board.result()}")
    print(f"Game {games + 1} completed.")

model.save("./models/chess-rl.h5")
engine.quit()
