import tensorflow as tf
import matplotlib.pyplot as plt

def load_and_preprocess_audio(file_path, label):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    return audio, label

def create_mel_spectrogram(audio_data, label, sample_rate=8000, frame_length=256, 
                           frame_step=128, num_mel_bins=64, lower_freq=30, upper_freq=2000, 
                           target_time_frames=250, augment=False):
    stft = tf.signal.stft(
        audio_data,
        frame_length=frame_length,
        frame_step=frame_step,
        pad_end=True
    )

    spectrogram = tf.abs(stft)
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_freq,
        upper_edge_hertz=upper_freq
    )
    
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = tf.image.resize_with_crop_or_pad(mel_spectrogram, target_time_frames, num_mel_bins)


    if augment:
        noise = tf.random.normal(tf.shape(mel_spectrogram), mean=0.0, stddev=0.2)
        mel_spectrogram = mel_spectrogram + noise
        
        # 2. Time mask 
        time_mask_param = 10
        time_max = target_time_frames
        t = tf.random.uniform([], minval=0, maxval=time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=time_max - t + 1, dtype=tf.int32)
        time_mask = tf.concat([
            tf.ones([t0, num_mel_bins, 1]),
            tf.zeros([t, num_mel_bins, 1]),
            tf.ones([time_max - t0 - t, num_mel_bins, 1])
        ], axis=0)
        mel_spectrogram = mel_spectrogram * time_mask

        # 3. Frequency mask
        freq_mask_param = 10
        freq_max = num_mel_bins
        f = tf.random.uniform([], minval=0, maxval=freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=freq_max - f + 1, dtype=tf.int32)
        # Create a mask along the frequency axis (axis=1)
        freq_mask = tf.concat([
            tf.ones([target_time_frames, f0, 1]),
            tf.zeros([target_time_frames, f, 1]),
            tf.ones([target_time_frames, freq_max - f0 - f, 1])
        ], axis=1)
        mel_spectrogram = mel_spectrogram * freq_mask
    
    return mel_spectrogram, label

def evaluate_model(history):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    f1_score = history.history.get("f1_score", [])
    val_f1_score = history.history.get("val_f1_score", [])
    auprc = history.history.get("auprc", [])
    val_auprc = history.history.get("val_auprc", [])
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, f1_score, "b-", label="Training f1_score")
    plt.plot(epochs, val_f1_score, "r-", label="Validation f1_score")
    plt.title("Training and Validation f1_score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 score")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, auprc, "b-", label="Training AUPRC")
    plt.plot(epochs, val_auprc, "r-", label="Validation AUPRC")
    plt.title("Training and Validation AUPRC")
    plt.xlabel("Epochs")
    plt.ylabel("AUPRC")
    plt.legend()

    plt.tight_layout()
    plt.show()

    best_val_f1 = tf.reduce_max(tf.constant(val_f1_score))
    best_val_auprc = tf.reduce_max(tf.constant(val_auprc))
    
    print(f"Best Validation F1 Score: {best_val_f1.numpy():.4f}")
    print(f"Best Validation AUPRC: {best_val_auprc.numpy():.4f}")
