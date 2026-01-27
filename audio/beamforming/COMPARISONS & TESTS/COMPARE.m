function val = compute_si_sdr(y, s)
% Scale-Invariant SDR

s = s(:); y = y(:);

s_proj = (dot(y,s) / dot(s,s)) * s;
e_noise = y - s_proj;

val = 10*log10( dot(s_proj,s_proj) / dot(e_noise,e_noise) );
end

function val = compute_osinr(y, s)
% Output SINR using orthogonal decomposition

s = s(:); y = y(:);

% Project output onto target
s_hat = (dot(y,s) / dot(s,s)) * s;

% Everything else = interference + noise
i_hat = y - s_hat;

val = 10*log10( dot(s_hat,s_hat) / dot(i_hat,i_hat) );
end


clc; clear;

%% ------------------- INPUTS -------------------
% Required signals (column vectors)
% s        : clean target reference
% y_mic1  : single mic output
% y_mvdr_taki   : delay-and-sum output
% y_mvdr  : MVDR output
% fs      : sampling rate

[s, fs] = audioread("/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/male_clean_15s.wav");
[y_mic1, ~] = audioread("/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_audio/multichannel_maleClean_femalePiano.wav");
[y_mvdr, ~] = audioread("/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/COMPARE2_V2_mvdr_emon_output.wav");
[y_mvdr_taki, ~] = audioread("/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Test_output/COMPARE2_V2_mvdr_taki_output.wav");


y_mic1 = y_mic1(:,1);

signals = {y_mic1, y_mvdr_taki, y_mvdr};
names   = {'Single Mic', 'MVDR_Taki', 'MVDR'};

numSys = length(signals);

SI_SDR  = zeros(numSys,1);
STOI_v  = zeros(numSys,1);
OSINR_v = zeros(numSys,1);

%% ------------------- METRICS -------------------
for k = 1:numSys
    y = signals{k};

    % Ensure same length
    L = min(length(s), length(y));
    s0 = s(1:L);
    y0 = y(1:L);

    %% ---- SI-SDR ----
    SI_SDR(k) = compute_si_sdr(y0, s0);

    %% ---- STOI ----
    STOI_v(k) = stoi(s0, y0, fs);

    %% ---- OSINR ----
    OSINR_v(k) = compute_osinr(y0, s0);
end

%% ------------------- DISPLAY -------------------
T = table(SI_SDR, STOI_v, OSINR_v, ...
          'RowNames', names, ...
          'VariableNames', {'SI_SDR_dB','STOI','OSINR_dB'});

disp(T);
