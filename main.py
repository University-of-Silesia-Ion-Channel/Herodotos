import pandas as pd
from herodotos import Herodotos
import pickle

if __name__ == '__main__':
    with open('data/simulation_m20_D0.23713737056616552.p', 'rb') as f:
        data = pickle.load(f)

    herodotos = Herodotos(data, bins=50)

    df = pd.DataFrame(data['x'], columns=['x'])
    print(df.head())

    # herodotos.naive()
    herodotos.deep_learning("./models/DeepChannel.keras")

    # herodotos.save("histogram.csv")

    # herodotos.plot(1400, 800)
