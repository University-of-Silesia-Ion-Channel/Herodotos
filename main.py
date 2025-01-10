import herodotos
from herodotos import Herodotos
import pickle

if __name__ == '__main__':
    with open('data/simulation_m20_D0.23713737056616552.p', 'rb') as f:
        data = pickle.load(f)

    herodotos = Herodotos(data)

    herodotos.naive()

    herodotos.plot()
