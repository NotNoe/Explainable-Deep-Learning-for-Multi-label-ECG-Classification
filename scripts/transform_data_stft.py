import h5py
from tqdm import tqdm
from scripts.Transformations import Transformations

destiny_dataset = "stft"
BATCH = 1000
files = ["data/train.hdf5", "data/test.hdf5", "data/validation.hdf5"]
for filename in files:
    with h5py.File(filename, 'r+') as file:
        ecgs = file['tracings']
        f, t = Transformations(ecgs[0:1]).get_stft_arrays()
        n_ecgs = ecgs.shape[0]
        #If the dataset exists, we delete it
        if destiny_dataset in file:
            del file[destiny_dataset]
        procesed = file.create_dataset(destiny_dataset, shape=(n_ecgs, len(f) * 12, len(t)), dtype='float32')
        for i in tqdm(range(0, n_ecgs, BATCH), "Processing file " + filename):
            end = min(i + BATCH, n_ecgs)
            batch_ecgs = ecgs[i:end,:,:]
            transformations = Transformations(batch_ecgs)
            transformed_ecg = transformations.stft()
            procesed[i:end] = transformed_ecg


            