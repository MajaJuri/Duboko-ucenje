import Zadatak3
import Zadatak2

# usporedba ćelija
def usporedba_RNN_celija_1():
    seeds = [7052020]  # neki seedovi
    rnn_cells = ['GRU', 'LSTM', 'Vanilla']
    for cell in rnn_cells:
        print("Model", cell)
        for (s, seed) in enumerate(seeds):
            print("Seed {}/{}: {}".format(s+1, len(seeds), seed))
            args = {
                'seed': seed,
                'min_freq': 1,
                'max_size': -1,
                'learning_rate': 1e-4,
                'batch_size_train': 10,
                'batch_size_test_validate': 32,
                'max_epochs': 5,
                'clip': 0.25,
                'rnn_cell': cell,
                'num_layers': 2,
                'hidden_size': 150,
                'dropout': 0,
                'bidirectional': False,
                'prednauceneReprezentacije': True
                }
            Zadatak3.main(args)


def usporedba_RNN_celija_2():
    seeds = [7052020]  # neki seedovi
    rnn_cells = ['GRU', 'LSTM', 'Vanilla']
    dropouts = [0.4, 0.8]
    bidirectionals = [True, False]
    hidden_sizes = [50, 200]
    num_layers = [3, 6]
    for cell in rnn_cells:
        for (i, num_layer) in enumerate(num_layers):
            for (k, hidden_size) in enumerate(hidden_sizes):
                for (j, dropout) in enumerate(dropouts):
                    for bidirectional in bidirectionals:
                        print("Model", cell)
                        print("Num layers {}/{}: {}".format(i + 1, len(num_layers), num_layer))
                        print("Hidden size {}/{}: {}".format(k + 1, len(hidden_sizes), hidden_size))
                        print("Dropout {}/{}: {}".format(j + 1, len(dropouts), dropout))
                        print("Bidirectional: {}".format(bidirectional))
                        for (s, seed) in enumerate(seeds):
                            print("Seed {}/{}: {}".format(s+1, len(seeds), seed))
                            args = {
                                'seed': seed,
                                'min_freq': 1,
                                'max_size': -1,
                                'learning_rate': 1e-4,
                                'batch_size_train': 10,
                                'batch_size_test_validate': 32,
                                'max_epochs': 5,
                                'clip': 0.25,
                                'rnn_cell': cell,
                                'num_layers': num_layer,
                                'hidden_size': hidden_size,
                                'dropout': dropout,
                                'bidirectional': bidirectional,
                                'prednauceneReprezentacije': True
                            }
                            Zadatak3.main(args)


def usporedba_RNN_celija_3():
    seeds = [7052020, 1256437, 3452674, 9854326, 8543219]  # neki seedovi
    for (i, seed) in enumerate(seeds):
        print("Seed {}/{}: {}".format(i + 1, len(seeds), seed))
        args = {
            'seed': seed,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 3,
            'hidden_size': 200,
            'dropout': 0.4,
            'bidirectional': True,
            'prednauceneReprezentacije': True
        }
        Zadatak3.main(args)


def prednauceneReprezentacije():
    seeds = [7052020]  # neki seedovi
    for (i, seed) in enumerate(seeds):
        for (j, prednaucene) in enumerate([True, False]):
            print("Seed {}/{}: {}".format(i + 1, len(seeds), seed))
            print('RNN model:')
            print("Prednaucene reprezentacije: ", prednaucene)
            args = {
                'seed': seed,
                'min_freq': 1,
                'max_size': -1,
                'learning_rate': 1e-4,
                'batch_size_train': 10,
                'batch_size_test_validate': 32,
                'max_epochs': 5,
                'clip': 0.25,
                'rnn_cell': 'LSTM',
                'num_layers': 3,
                'hidden_size': 200,
                'dropout': 0.4,
                'bidirectional': True,
                'prednauceneReprezentacije': prednaucene
            }
            Zadatak3.main(args)

            print("Basic model:")
            print("Prednaucene reprezentacije: ", prednaucene)
            args = {
                'seed': seed,
                'min_freq': 1,
                'max_size': -1,
                'learning_rate': 1e-4,
                'batch_size_train': 10,
                'batch_size_test_validate': 32,
                'max_epochs': 5,
                'prednauceneReprezentacije': prednaucene
            }
            Zadatak2.main(args)


def hiperparametriZvjezdice():
    print("PARAMETAR MIN FREQ")
    for m in [1, 10, 20]:
        print("min_freq = ", m)
        print('RNN model:')
        args = {
            'seed': 7052020,
            'min_freq': m,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 3,
            'hidden_size': 200,
            'dropout': 0.4,
            'bidirectional': True,
            'prednauceneReprezentacije': False
        }
        Zadatak3.main(args)

        print("Basic model:")
        args = {
            'seed': 7052020,
            'min_freq': m,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'prednauceneReprezentacije': True
        }
        Zadatak2.main(args)

    print("PARAMETAR LEARNING RATE")
    for l in [1e-4, 1e-2, 1]:
        print("learning rate = ", l)
        print('RNN model:')
        args = {
            'seed': 7052020,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': l,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 3,
            'hidden_size': 200,
            'dropout': 0.4,
            'bidirectional': True,
            'prednauceneReprezentacije': False
        }
        Zadatak3.main(args)

        print("Basic model:")
        args = {
            'seed': 7052020,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': l,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'prednauceneReprezentacije': True
        }
        Zadatak2.main(args)

    print("PARAMETAR DROPOUT")
    for d in [0, 0.4, 0.8]:
        print("dropout = ", d)
        print('RNN model:')
        args = {
            'seed': 7052020,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 3,
            'hidden_size': 200,
            'dropout': d,
            'bidirectional': True,
            'prednauceneReprezentacije': False
        }
        Zadatak3.main(args)

    print("PARAMETAR BROJ SLOJEVA")
    for n in [2, 5, 8]:
        print("broj slojeva = ", n)
        print('RNN model:')
        args = {
            'seed': 7052020,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': n,
            'hidden_size': 200,
            'dropout': 0.4,
            'bidirectional': True,
            'prednauceneReprezentacije': False
        }
        Zadatak3.main(args)

    print("PARAMETAR FREEZE")
    for f in [True, False]:
        print("freeze = ", f)
        print('RNN model:')
        args = {
            'seed': 7052020,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 3,
            'hidden_size': 200,
            'dropout': 0.4,
            'bidirectional': True,
            'prednauceneReprezentacije': f
        }
        Zadatak3.main(args)

        print("Basic model:")
        args = {
            'seed': 7052020,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'prednauceneReprezentacije': f
        }
        Zadatak2.main(args)


def najboljiParametri():
    seeds = [7052020, 1256437, 3452674, 9854326, 8543219]  # neki seedovi
    for (i, seed) in enumerate(seeds):
        print("Seed {}/{}: {}".format(i + 1, len(seeds), seed))
        print("RNN model:")
        # najbolji parametri:
        # min_freq = 1 (iako za sve dode isti F1)
        # learning_rate = 1e-4
        # dropout = 0
        # broj slojeva = 2
        # freeze = True
        args = {
            'seed': seed,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 2,
            'hidden_size': 200,
            'dropout': 0,
            'bidirectional': True,
            'prednauceneReprezentacije': True
        }
        Zadatak3.main(args)

        print("Basic model:")
        # najbolji parametri:
        # min_freq = 1 (iako za sve dode isti F1)
        # learning_rate = 1e-4
        # freeze = True
        args = {
            'seed': seed,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'prednauceneReprezentacije': True
        }
        Zadatak2.main(args)


if __name__ == "__main__":
    #usporedba_RNN_celija_1()
    #usporedba_RNN_celija_2()
    #usporedba_RNN_celija_3()
    #prednauceneReprezentacije()
    #hiperparametriZvjezdice()
    najboljiParametri()

