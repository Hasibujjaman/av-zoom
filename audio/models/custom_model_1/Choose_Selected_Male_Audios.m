% === SET ROOT DIRECTORY EXPLICITLY ===
ROOT = '/Users/emonchowdhury/Desktop/Phase 2/av_zoom/audio/models/custom_model_1';   % <<< CHANGE THIS PATH

cd(ROOT);

post_mvdr_dir = fullfile(ROOT, 'mvdr_outputs');
male_dir      = fullfile(ROOT, 'male_clean');
out_dir       = fullfile(ROOT, 'true_labels');

% Create output directory if it doesn't exist
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Created output directory: %s\n', out_dir);
end

% ===== LIST FILES =====
post_mvdr_files = dir(fullfile(post_mvdr_dir, '*.wav'));
maleFiles  = dir(fullfile(male_dir, '*.flac'));

% Check if directories exist
if isempty(post_mvdr_files)
    error('No .wav files found in %s', post_mvdr_dir);
end

if isempty(maleFiles)
    error('No .flac files found in %s', male_dir);
end

fprintf('Found %d post-MVDR files and %d male clean files.\n', ...
    length(post_mvdr_files), length(maleFiles));

% ===== EXTRACT UNIQUE BASE IDS FROM POST-MVDR FILES =====
usedIDs = containers.Map('KeyType', 'char', 'ValueType', 'logical');
extractedCount = 0;

for i = 1:length(post_mvdr_files)
    fname = post_mvdr_files(i).name;
    
    % Example pattern: 8758_part2_E_female_music_noise.wav
    % Extract "8758_part2" (everything before _[A-E]_)
    tokens = regexp(fname, '^(.+?)_[A-E]_', 'tokens');
    
    if ~isempty(tokens)
        baseID = tokens{1}{1};
        if ~isKey(usedIDs, baseID)
            usedIDs(baseID) = true;
            extractedCount = extractedCount + 1;
        end
    else
        % Alternative pattern if the first one doesn't match
        % Try to extract just the numeric prefix
        tokens = regexp(fname, '^(\d+)_', 'tokens');
        if ~isempty(tokens)
            baseID = tokens{1}{1};
            if ~isKey(usedIDs, baseID)
                usedIDs(baseID) = true;
                extractedCount = extractedCount + 1;
            end
        end
    end
end

fprintf('Found %d unique IDs from post-MVDR files.\n', usedIDs.Count);

% ===== COPY MATCHING MALE FILES =====
copied = 0;
skipped = 0;

for i = 1:length(maleFiles)
    % Get base name without extension
    [~, maleID, ~] = fileparts(maleFiles(i).name);
    
    % Remove trailing "_clean" if present (common in clean audio files)
    maleID = regexprep(maleID, '_clean$', '');
    
    if isKey(usedIDs, maleID)
        src = fullfile(male_dir, maleFiles(i).name);
        dst = fullfile(out_dir, maleFiles(i).name);
        
        % Check if file already exists in destination
        if exist(dst, 'file')
            fprintf('Skipping %s (already exists in output directory)\n', maleFiles(i).name);
            skipped = skipped + 1;
            continue;
        end
        
        % Check if source file exists
        if ~exist(src, 'file')
            fprintf('Warning: Source file %s does not exist\n', src);
            continue;
        end
        
        try
            copyfile(src, dst);
            copied = copied + 1;
            fprintf('Copied: %s\n', maleFiles(i).name);
        catch ME
            fprintf('Error copying %s: %s\n', maleFiles(i).name, ME.message);
        end
    end
end

fprintf('\n===== SUMMARY =====\n');
fprintf('Total post-MVDR files processed: %d\n', length(post_mvdr_files));
fprintf('Unique IDs extracted: %d\n', usedIDs.Count);
fprintf('Male clean files available: %d\n', length(maleFiles));
fprintf('Files copied to %s: %d\n', out_dir, copied);
fprintf('Files skipped (already exist): %d\n', skipped);

if copied == 0
    fprintf('\nWarning: No files were copied!\n');
    fprintf('Possible issues:\n');
    fprintf('1. Filename patterns might not match\n');
    fprintf('2. Check regex pattern in line 28\n');
    fprintf('3. Verify male clean files have matching base IDs\n');
    
    % Display some sample IDs for debugging
    fprintf('\nSample post-MVDR IDs (first 5):\n');
    usedIDs_keys = keys(usedIDs);
    for k = 1:min(5, length(usedIDs_keys))
        fprintf('  %s\n', usedIDs_keys{k});
    end
    
    fprintf('\nSample male file IDs (first 5):\n');
    for k = 1:min(5, length(maleFiles))
        [~, name, ~] = fileparts(maleFiles(k).name);
        fprintf('  %s\n', regexprep(name, '_clean$', ''));
    end
end