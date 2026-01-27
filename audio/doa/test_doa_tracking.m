clc; clear;

fs = 16000;
mic_dist = 0.08;

% Simulated moving source
theta_true = linspace(-60, 60, 200);
tau_true = mic_dist * sind(theta_true) / 343;

% Generate signals
N = 1024;
x1 = randn(N,1);
x2 = zeros(N,1);

for k = 1:length(tau_true)
    delay = round(tau_true(k) * fs);
    x2 = circshift(x1, delay);

    X = fft([x1 x2]);
    theta_est(k) = estimate_doa(X, fs, mic_dist);
end

plot(theta_true, 'k'); hold on;
plot(theta_est, 'r--');
legend('True', 'Estimated');
xlabel('Frame'); ylabel('Angle (deg)');
