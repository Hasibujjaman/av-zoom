%% ============================================================
%  SP CUP 2026 – FINAL CLEAN Mixture Generator (Pure ISM)
%% ============================================================

clear; clc; close all;


num_male_speech = 2;

%% ---------------- Parameters ----------------
fs = 16000;
c = 340;
roomDim = [4.9 4.9 4.9];
RT60 = 0.5;
order = 6;
SIR_dB = 0;
SNR_dB = 5;
T = 3; 
N = T*fs;

%% ---------------- Geometry ----------------
micPos = [2.41 2.45 1.5;
          2.49 2.45 1.5];

src_target = [2.45 3.45 1.5];
src_interf = [3.22 3.06 1.5];

%% ---------------- Paths ----------------
root = "/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Dataset Generation";

maleDir   = fullfile(root,"Male_clean");
femaleDir = fullfile(root,"Female");
musicDir  = fullfile(root,"Music");
noiseDir  = fullfile(root,"Noise");

outDir = fullfile(root,"Generated_Mixtures");
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ---------------- Files ----------------
maleFiles   = dir(fullfile(maleDir,"*.flac"));
femaleFiles = dir(fullfile(femaleDir,"*.flac"));
musicFiles  = dir(fullfile(musicDir,"*.flac"));
noiseFiles  = dir(fullfile(noiseDir,"*.flac"));

combos = {"female","music","noise","female_music","female_noise"};

%% ---------------- Absorption ----------------
V = prod(roomDim);
S = 2*(roomDim(1)*roomDim(2)+roomDim(1)*roomDim(3)+roomDim(2)*roomDim(3));
alpha = min(0.99, 0.161*V/(RT60*S));
beta = sqrt(1-alpha);

fprintf("Using reflection coefficient beta = %.3f\n", beta);

%% ================= MAIN LOOP =================
for m = 1:min(num_male_speech, length(maleFiles)) 
    maleFile = maleFiles(m).name;
    [maleSig,~] = audioread(fullfile(maleDir,maleFile));
    maleSig = maleSig(1:N);
    maleSig = maleSig / rms(maleSig);

    baseName = erase(maleFile,".flac");
    fprintf("\nMale file: %s\n", maleFile);

    %% RIRs
    rir_t = ism_rir(roomDim, micPos, src_target, fs, c, beta, order);
    rir_i = ism_rir(roomDim, micPos, src_interf, fs, c, beta, order);

    fprintf("RT60 target RIR = %.3f s | interf = %.3f s\n", ...
        rt60_schroeder(rir_t(:,1),fs), rt60_schroeder(rir_i(:,1),fs));

    for k = 1:length(combos)
        combo = combos{k};
        fprintf("  Combo: %s\n", combo);

        %% ---------- Select interference ----------
        switch combo
            case "female"
                interf = audioread(fullfile(femaleFiles(randi(end)).folder,...
                                            femaleFiles(randi(end)).name));
                % %%% 
                % interf = audioread('/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/beamforming/Dataset Generation/Female/2_part22.flac');
                % %%%
            case "music"
                interf = audioread(fullfile(musicFiles(randi(end)).folder,...
                                            musicFiles(randi(end)).name));
            case "noise"
                interf = audioread(fullfile(noiseFiles(randi(end)).folder,...
                                            noiseFiles(randi(end)).name));
            case "female_music"
                interf = audioread(fullfile(femaleFiles(randi(end)).folder,...
                                            femaleFiles(randi(end)).name)) + ...
                         audioread(fullfile(musicFiles(randi(end)).folder,...
                                            musicFiles(randi(end)).name));
            case "female_noise"
                interf = audioread(fullfile(femaleFiles(randi(end)).folder,...
                                            femaleFiles(randi(end)).name)) + ...
                         audioread(fullfile(noiseFiles(randi(end)).folder,...
                                            noiseFiles(randi(end)).name));
        end

        interf = interf(1:N);
        interf = interf / rms(interf);

        %% ---------- SIR = 0 dB ----------
        interf = interf * rms(maleSig)/rms(interf);

        %% ---------- Convolution ----------
        tgt_mc = zeros(N,2);
        int_mc = zeros(N,2);

        for mic=1:2
            tgt_mc(:,mic) = fftfilt(rir_t(:,mic), maleSig);
            int_mc(:,mic) = fftfilt(rir_i(:,mic), interf);
        end

        %% ---------- Normalize AFTER convolution ----------
        scale = max(abs([tgt_mc(:); int_mc(:)]));
        tgt_mc = tgt_mc / scale;
        int_mc = int_mc / scale;

        mixture_clean = tgt_mc + int_mc;

        %% ---------- Add AWGN ----------
        noise = randn(size(mixture_clean));
        for mic=1:2
            Ps = mean(mixture_clean(:,mic).^2);
            Pn = Ps / (10^(SNR_dB/10));
            noise(:,mic) = noise(:,mic)*sqrt(Pn/mean(noise(:,mic).^2));
        end

        mixture = mixture_clean + noise;

        %% ---------- Final safety normalization ----------
        mixture = 0.99 * mixture / max(abs(mixture(:)));

        %% ---------- Verify ----------
        for mic=1:2
            fprintf("    Mic %d | SIR=%.2f dB | SNR=%.2f dB\n", mic, ...
                10*log10(mean(tgt_mc(:,mic).^2)/mean(int_mc(:,mic).^2)), ...
                10*log10(mean(mixture_clean(:,mic).^2)/mean(noise(:,mic).^2)));
        end

        %% ---------- Save ----------
        outName = sprintf("%s_%s.wav", baseName, combo);
        audiowrite(fullfile(outDir,outName), mixture, fs);
    end
end

disp("✅ Mixture generation COMPLETE.");




%% ================= FUNCTIONS =================
function h = ism_rir(room, mics, src, fs, c, beta, order)
Nm = size(mics,1);
h = zeros(round(fs*1.5), Nm);

for nx=-order:order
for ny=-order:order
for nz=-order:order
    refl = beta^(abs(nx)+abs(ny)+abs(nz));
    img = [(-1)^nx*src(1)+2*nx*room(1), ...
           (-1)^ny*src(2)+2*ny*room(2), ...
           (-1)^nz*src(3)+2*nz*room(3)];
    for m=1:Nm
        d = norm(img - mics(m,:));
        t = round(d/c*fs)+1;
        if t<=size(h,1)
            h(t,m) = h(t,m) + refl/d;
        end
    end
end
end
end
end

function rt = rt60_schroeder(h,fs)
h = h.^2;
edc = flipud(cumsum(flipud(h)));
edc = edc/max(edc);
edc_db = 10*log10(edc+eps);

i1 = find(edc_db<-5,1);
i2 = find(edc_db<-35,1);
rt = (i2-i1)/fs*2;
end
