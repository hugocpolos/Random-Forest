from src.Dataset import Dataset
from src.DecisionTree import DecisionTree
from src.Forest import Forest
from sys import argv
from src.Classifier import Classifier


def print_usage(bin_name):
    print("""Usage:
        %s [Arguments]

        - Arguments:
            [--file, -f]        csv filename
            [--delimiter, -d]   csv delimiter character,
                                the default value is ';'
            [--meta, -m]        json metadata filename
            [--seed, -s]        random for pseudo-random number generation,
                                random value if a seed is not set
            [--length, -l, -m]  forent-length, the default value is 1
            [--print, -p]       print the forest

        """ % (bin_name))


if __name__ == '__main__':
    if len(argv) is 1:
        print_usage(argv[0])
        exit(0)
    else:
        filename = ""
        delimiter = ';'
        metadata = ""
        seed = None
        n = 1
        to_print = False
        for i in range(len(argv)):
            if(argv[i] in ['--file', '-f']):
                filename = argv[i + 1]
            elif(argv[i] in ['--delimiter', '-d']):
                delimiter = argv[i + 1]
            elif(argv[i] in ['--meta', '-m']):
                metadata = argv[i + 1]
            elif(argv[i] in ['--seed', '-s']):
                seed = int(argv[i + 1])
            elif(argv[i] in ['--length', '-l', '-n']):
                n = int(argv[i + 1])
            elif(argv[i] in ['--print', '-p']):
                to_print = True
            elif(argv[i] in ['--help', '-h']):
                print_usage(argv[0])
                exit(0)

        # load the dataset to memory
        db = Dataset(filename, delimiter=delimiter,
                     metadata=metadata)

        for i in range(1, n + 1):
            # Load the Forest Object
            F = Forest(db, forest_length=i, seed=seed)

            # Train the Forest
            F.train()
            if(to_print):
                F.show()

            print("n = %d, hit ratio :" % (i), end=' ')

            # Test the Forest
            F.test()

        exit(0)
