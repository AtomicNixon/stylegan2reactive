import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import librosa
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

audio_file = 'beat.wav'
# network = 'ffhq.pkl'
network = 'testn1.pkl'

def main():
    y, sr = librosa.load(audio_file)
    with open(network, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    device = torch.device('cuda')

    # Calculate the hop length for a time interval of 1/30 seconds
    interval = 1/30  # desired time interval in seconds
    hop_length = int(sr * interval)  # hop length in samples

    # Compute a short-time Fourier transform (STFT) to use as the basis for onset detection
    D = np.abs(librosa.stft(y, hop_length=hop_length))

    # Compute the spectrogram
    spectrogram = librosa.amplitude_to_db(D, ref=np.max)

    # Define the range of frequencies considered as 'treble' (e.g., 2000 to 5000 Hz)
    r1_low = 580
    r1_high = 650
    r2_low = 50
    r2_high = 150

    # Convert the frequency bounds to spectrogram bin indices
    r1_low_bin = int(r1_low * spectrogram.shape[0] / (sr / 2))
    r1_high_bin = int(r1_high * spectrogram.shape[0] / (sr / 2))
    r2_low_bin = int(r2_low * spectrogram.shape[0] / (sr / 2))
    r2_high_bin = int(r2_high * spectrogram.shape[0] / (sr / 2))

    # Sum the spectrogram intensities in the 'treble' range for each frame
    r1_volume = np.sum(spectrogram[r1_low_bin:r1_high_bin, :], axis=0)
    r2_volume = np.sum(spectrogram[r2_low_bin:r2_high_bin, :], axis=0)
    # Normalize the treble volume to the range [0, 1]
    normalized_r1_volume = (r1_volume - np.min(r1_volume)) / (np.max(r1_volume) - np.min(r1_volume))
    normalized_r2_volume = (r2_volume - np.min(r2_volume)) / (np.max(r2_volume) - np.min(r2_volume))

    seed = 1007
    psi = 0.8

    # Generate latent vector
    torch.manual_seed(seed)
    z = torch.randn([1, G.z_dim], device='cuda')  # Latent vector (z) with random values
#    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    # Truncation trick
    w_avg = G.mapping.w_avg
    w = G.mapping(z, None)
    w = w_avg + psi * (w - w_avg)

    def normalize_vector(v):
        v_np = v.cpu().numpy()  # Convert the tensor to a NumPy array
        w_avg_np = w_avg.cpu().numpy()  # Convert the tensor to a NumPy array
        return v_np * np.std(w_avg_np) / np.std(v_np) + np.mean(w_avg_np) - np.mean(v_np)

    mouth_open_vector = torch.from_numpy(np.load('vectors/mouth_open.npy')).to('cuda')
    mouth_open = normalize_vector(mouth_open_vector)
    mouth_open_tensor = torch.from_numpy(mouth_open).to('cuda')

    age_vector = torch.from_numpy(np.load('vectors/mouth_size.npy')).to('cuda')
    age = normalize_vector(age_vector)
    age_tensor = torch.from_numpy(age).to('cuda')

    gender_vector = torch.from_numpy(np.load('vectors/smile.npy')).to('cuda')
    gender = normalize_vector(gender_vector)
    gender_tensor = torch.from_numpy(gender).to('cuda')

    r1 = (normalized_r1_volume)
    r2 = (normalized_r2_volume)

    print('r1='+str(np.min(r1))+'  '+str(np.max(r1)))
    print('r2='+str(np.min(r2))+'  '+str(np.max(r2)))
    frames = len(r1_volume)
    print(str(len(r1)))

    images = []
    seeds = [3, 9, 16, 28, 31, 33, 44, 68, 66, 76, 55, 32] # List of seeds to interpolate between
    num_frames_per_transition = 150  # Number of frames for each transition
  
    # Generate the latent vectors for each seed
    zs = [torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device) for seed in seeds]

    # Map each z to w
    ws = [G.mapping(z, None) for z in zs]

    frame_number = 0  # Keep track of the frame number

    # Interpolate between each pair of w's
    for i in range(len(ws) - 1):
#    for i in range(len(r1) - 1):
        w1 = ws[i]
        w2 = ws[i + 1]
        for t in np.linspace(0, 1, num_frames_per_transition):
            w = (1 - t) * w1 + t * w2  # Linear interpolation

            w2 = w
            w2 = w2 + mouth_open_tensor * r1[frame_number]
            w2 = w2 + age_tensor * r2[frame_number]
            w2 = w2 + gender_tensor * r2[frame_number]
            with torch.no_grad():
                img = G.synthesis(w2, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            img_np = np.array(img) 
            images.append(img_np)

            frame_number += 1  # Increment the frame number

    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile("output.mp4", codec='mpeg4')

if __name__ == "__main__":
    main()
