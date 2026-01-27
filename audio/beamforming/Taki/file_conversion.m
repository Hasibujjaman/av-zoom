[y, fs] = audioread('840973__disquiet__two-block-drone.wav');
audiowrite('target_signal.wav', y, fs);
