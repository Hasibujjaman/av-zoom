addpath('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming');

% Parameters
fs = 16000;
f0 = 1000;            % test frequency
c = 343;
d = 0.06;             % mic spacing (m)
theta = 30;           % DOA (degrees)

% Simulated mic signals (complex)
X = [1;
     exp(-1j*2*pi*f0*(d*sind(theta)/c))] ...
     + 0.1*(randn(2,1)+1j*randn(2,1));

% Steering vector
tau = [0; d*sind(theta)/c];
a = exp(-1j*2*pi*f0*tau);

% Covariance
Rxx = X * X';

% Diagonal loading (important!)
epsilon = 1e-3;
Rxx = Rxx + epsilon*eye(2);

% MVDR weights
w = (Rxx \ a) / (a'*(Rxx\a));

% Beamformer output
Y = w' * X;

disp("MVDR Output:");
disp(Y);


% Test1
disp("Test 1");
constraint_value = w' * a;
disp("Distortionless constraint (should be ~1):");
disp(constraint_value);

% Test 2
disp("Test 2");
y_single = X(1);   % mic 1 only

disp("Single mic power:");
disp(abs(y_single)^2);

disp("MVDR output power:");
disp(abs(Y)^2);


% Test 3
disp("Test 3");
theta_wrong = -30;   % wrong direction
tau_wrong = [0; d*sind(theta_wrong)/c];
a_wrong = exp(-1j*2*pi*f0*tau_wrong);

w_wrong = (Rxx \ a_wrong) / (a_wrong'*(Rxx\a_wrong));
Y_wrong = w_wrong' * X;

disp("Correct DOA output magnitude:");
disp(abs(Y));

disp("Wrong DOA output magnitude:");
disp(abs(Y_wrong));

% Test 4
disp("Test 4");
theta0 = 0;
tau0 = [0; d*sind(theta0)/c];
a0 = exp(-1j*2*pi*f0*tau0);

w0 = (Rxx \ a0) / (a0'*(Rxx\a0));

disp("Weights at 0 deg DOA:");
disp(w0);

% Visual Test
angles = -90:1:90;
response = zeros(size(angles));

for k = 1:length(angles)
    tau_k = [0; d*sind(angles(k))/c];
    a_k = exp(-1j*2*pi*f0*tau_k);
    response(k) = abs(w' * a_k);
end

figure;
plot(angles, response);
xlabel("Angle (deg)");
ylabel("Beam response");
title("MVDR Beampattern (Single Frequency)");
grid on;
