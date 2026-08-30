% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% =========================================================================
% Paper Section 7.2.2, Test 2: Streaming CT with 1000x1000 phantom.
% 6 random subproblems (30 random angles each).
% =========================================================================
clc
clear all
close all
rng(17, 'v4');

%% ---- Setup paths ----
directory = pwd;
addpath(directory)
addpath(fullfile(directory, 'utilities'))
addpath(fullfile(directory, 'AIRToolsII-master'))
AIRToolsII_setup
addpath(fullfile(directory, 'IRTools'))
IRtools_setup

%% ---- Output directories ----
outdir = fullfile(pwd, 'results', 'Test2');
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ---- Parameters (matching paper) ----
N       = 1000;
q       = 1;
epsilon = 0.001;
kmin    = 10;
kmax    = 30;
s       = kmax - kmin;          
tol     = 1e-8;                
tol_stream = 1e-3;             
mmgks_iter = kmax;              
iter_lmmgks = 10;             
iter_per_sub_fixed = 3;        
max_outer_per_sub_tol = 5;      
nblocks = 6;

noise_levels = [0.001, 0.005];  

%% ========================================================================
%  GENERATE DATA
%  Generate one full system A with 180 angles, then randomly split rows into 6 subproblems.
% =========================================================================
fprintf('Generating clean streaming CT data (Test 2, 6 random subproblems)...\n');

options = PRtomo('defaults');
options.angles = linspace(0, 179, 180);  % 180 angles, 1 deg spacing
[A_full_base, b_full_clean, x_true_vec] = PRtomo(N, options);
b_full_clean = b_full_clean(:);
x_true = x_true_vec(:);
nx = N; ny = N;
n_angles = 180;
n_rays = size(A_full_base, 1) / n_angles;

[nrows_full, ~] = size(A_full_base);
angles_per_sub = floor(n_angles / nblocks);  
blocksize = angles_per_sub * n_rays;

rng(17, 'v4');
angle_perm = randperm(n_angles);

% Regularization operator
LI = build_L1(nx, ny, 1);

fprintf('  Image: %d x %d\n', N, N);
fprintf('  Full system: %d x %d\n', nrows_full, N*N);
fprintf('  %d subproblems: %d rows each\n', nblocks, blocksize);
fprintf('  kmin=%d, kmax=%d, s=%d\n\n', kmin, kmax, s);

%% ---- Storage for results ----
n_noise = length(noise_levels);
RRE_table = zeros(n_noise, 6);

%% ========================================================================
%  LOOP OVER NOISE LEVELS
% =========================================================================
for ni = 1:n_noise
    sigma = noise_levels(ni);
    fprintf('\n=========================================================================\n');
    fprintf('  NOISE LEVEL: sigma = %.1f%%\n', sigma*100);
    fprintf('=========================================================================\n');

    % ---- Add noise to full stacked data ----
    rng(17, 'v4');
    e_full = randn(size(b_full_clean));
    e_full = e_full / norm(e_full) * norm(b_full_clean) * sigma;
    b_full = b_full_clean + e_full;

    % ---- Split into 6 random subproblems by whole angles ----
    As = cell(1, nblocks);
    bs = cell(1, nblocks);
    for i = 1:nblocks
        sub_angles = sort(angle_perm(((i-1)*angles_per_sub + 1) : i*angles_per_sub));
        row_idx = [];
        for a = sub_angles
            row_idx = [row_idx, ((a-1)*n_rays + 1) : (a*n_rays)];
        end
        As{i} = A_full_base(row_idx, :);
        bs{i} = b_full(row_idx);
    end

    % ==== Method 1: MM-GKS 1st (30 iters) ====
    fprintf('  1. MM-GKS 1st (%d iters)...\n', mmgks_iter);
    rng(17, 'v4');
    [x_MMGKS1, ~, ~] = MMGKS(As{1}, bs{1}, LI, q, epsilon, mmgks_iter, tol, x_true);
    RRE_MMGKS1 = norm(x_MMGKS1(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_MMGKS1);

    % ==== Method 2: LM-MM-GKS 1st (200 iters) ====
    fprintf('  2. LM-MM-GKS 1st (%d iters)...\n', iter_lmmgks*s);
    rng(17, 'v4');
    [x_LMMGKS1, ~, ~, ~] = ...
        LMMGKS(As{1}, bs{1}, LI, q, epsilon, iter_lmmgks, s, kmin, tol, 'TSVD', x_true);
    RRE_LMMGKS1 = norm(x_LMMGKS1(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_LMMGKS1);

    % ==== Method 3: MM-GKS all (30 iters) ====
    fprintf('  3. MM-GKS all (%d iters)...\n', mmgks_iter);
    rng(17, 'v4');
    [x_MMGKSall, ~, ~] = MMGKS(A_full_base, b_full, LI, q, epsilon, mmgks_iter, tol, x_true);
    RRE_MMGKSall = norm(x_MMGKSall(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_MMGKSall);

    % ==== Method 4: LM-MM-GKS all (200 iters) ====
    fprintf('  4. LM-MM-GKS all (%d iters)...\n', iter_lmmgks*s);
    rng(17, 'v4');
    [x_LMMGKSall, ~, info_LMMGKSall, si_LMMGKSall] = ...
        LMMGKS(A_full_base, b_full, LI, q, epsilon, iter_lmmgks, s, kmin, tol, 'TSVD', x_true);
    RRE_LMMGKSall = norm(x_LMMGKSall(:) - x_true) / norm(x_true);
    rre_LMMGKSall = build_rre(si_LMMGKSall);
    [best_rre, best_iter] = min(rre_LMMGKSall);
    fprintf('    RRE = %.4f (best = %.4f at iter %d)\n', RRE_LMMGKSall, best_rre, best_iter);

    % ==== Method 5: s-LM-MM-GKS fixed (360 total) ====
    fprintf('  5. s-LM-MM-GKS (fixed, %d total)...\n', iter_per_sub_fixed*s*nblocks);
    rng(17, 'v4');
    V_recycle = [];
    x_sLMMGKS_subs = cell(1, nblocks);
    RRE_sLMMGKS_subs = zeros(1, nblocks);
    rre_sLMMGKS_hist = [];
    for p = 1:nblocks
        [x_sub, ~, info_sub, si_sub] = ...
            LMMGKS(As{p}, bs{p}, LI, q, epsilon, iter_per_sub_fixed, s, kmin, ...
            tol, 'TSVD', x_true, V_recycle);
        V_recycle = info_sub.oldV;
        x_sLMMGKS_subs{p} = x_sub;
        RRE_sLMMGKS_subs(p) = norm(x_sub(:) - x_true) / norm(x_true);
        rre_sLMMGKS_hist = [rre_sLMMGKS_hist; build_rre(si_sub)];
    end
    x_sLMMGKS = x_sub;
    RRE_sLMMGKS = RRE_sLMMGKS_subs(nblocks);
    fprintf('    RRE = %.4f\n', RRE_sLMMGKS);

    % ==== Method 6: s-LM-MM-GKS tol (tol=1e-3) ====
    fprintf('  6. s-LM-MM-GKS (tol=%.0e, max %d/sub)...\n', tol_stream, max_outer_per_sub_tol*s);
    rng(17, 'v4');
    V_recycle_tol = [];
    x_sLMMGKS_tol_subs = cell(1, nblocks);
    RRE_sLMMGKS_tol_subs = zeros(1, nblocks);
    rre_sLMMGKS_tol_hist = [];
    for p = 1:nblocks
        [x_sub_tol, ~, info_sub_tol, si_sub_tol] = ...
            LMMGKS(As{p}, bs{p}, LI, q, epsilon, max_outer_per_sub_tol, s, kmin, ...
            tol_stream, 'TSVD', x_true, V_recycle_tol);
        V_recycle_tol = info_sub_tol.oldV;
        x_sLMMGKS_tol_subs{p} = x_sub_tol;
        RRE_sLMMGKS_tol_subs(p) = norm(x_sub_tol(:) - x_true) / norm(x_true);
        rre_sLMMGKS_tol_hist = [rre_sLMMGKS_tol_hist; build_rre(si_sub_tol)];
    end
    RRE_sLMMGKS_tol = RRE_sLMMGKS_tol_subs(nblocks);
    fprintf('    RRE = %.4f\n', RRE_sLMMGKS_tol);
    RRE_table(ni, :) = [RRE_MMGKS1, RRE_LMMGKS1, RRE_MMGKSall, RRE_LMMGKSall, RRE_sLMMGKS, RRE_sLMMGKS_tol];

    % ==================================================================
    %  FIGURES (only for sigma = 0.1%)
    % ==================================================================
    if abs(sigma - 0.001) < 1e-6
        fprintf('\n  Generating figures for sigma = 0.1%%...\n');
        imgdir = fullfile(outdir, 'images');
        if ~exist(imgdir, 'dir'), mkdir(imgdir); end
        datdir = fullfile(outdir, 'data');
        if ~exist(datdir, 'dir'), mkdir(datdir); end

        clim_rec = [0, max(x_true)];
        rev_color = 1;

        % ---- True image ----
        figure('Visible','off'); imagesc(reshape(x_true,N,N), clim_rec); axis image off; colormap gray;
        set(gca,'Position',[0 0 1 1]);
        exportgraphics(gcf, fullfile(imgdir, 'fig_setup_true.jpg'), 'Resolution', 300); close

        % ---- Full sinogram ----
        figure('Visible','off');
        imagesc(reshape(b_full_clean, n_rays, n_angles)); axis image off; colormap gray;
        set(gca,'Position',[0 0 1 1]);
        exportgraphics(gcf, fullfile(imgdir, 'fig_setup_sino_full.jpg'), 'Resolution', 300); close

        % ---- Sinograms for 6 subproblems ----
        n_angles_sub = blocksize / n_rays;
        for si = 1:nblocks
            figure('Visible','off');
            imagesc(reshape(bs{si}, n_rays, n_angles_sub)); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(imgdir, sprintf('fig_setup_sino%d.jpg', si)), 'Resolution', 300); close
        end

        % ---- Reconstructions and errors ----
        rec_names = {'MMGKS_1st', 'LMMGKS_1st', 'MMGKS_all', 'LMMGKS_all', 'sLMMGKS'};
        methods_x = {x_MMGKS1, x_LMMGKS1, x_MMGKSall, x_LMMGKSall, x_sLMMGKS};
        for mi = 1:5
            figure('Visible','off'); imagesc(reshape(methods_x{mi}, N, N), clim_rec); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(imgdir, sprintf('rec_%s.jpg', rec_names{mi})), 'Resolution', 300); close

            figure('Visible','off');
            imagesc(reshape(methods_x{mi}(:)-x_true, N, N), [-rev_color, 0]); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(imgdir, sprintf('err_%s.jpg', rec_names{mi})), 'Resolution', 300); close
        end

        % ---- Streaming progression (tol version) ----
        for p = 1:nblocks
            figure('Visible','off'); imagesc(reshape(x_sLMMGKS_tol_subs{p}, N, N), clim_rec); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(imgdir, sprintf('rec_stream_sub%d.jpg', p)), 'Resolution', 300); close

            figure('Visible','off');
            imagesc(reshape(x_sLMMGKS_tol_subs{p}(:)-x_true, N, N), [-rev_color, 0]); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(imgdir, sprintf('err_stream_sub%d.jpg', p)), 'Resolution', 300); close
        end

        T = table((1:length(rre_sLMMGKS_hist))', rre_sLMMGKS_hist(:), 'VariableNames', {'Iteration','RRE'});
        writetable(T, fullfile(datdir, 'test2_sLMMGKS.dat'), 'Delimiter', '\t', 'FileType', 'text');

        T = table((1:length(rre_sLMMGKS_tol_hist))', rre_sLMMGKS_tol_hist(:), 'VariableNames', {'Iteration','RRE'});
        writetable(T, fullfile(datdir, 'test2_sLMMGKS_tol.dat'), 'Delimiter', '\t', 'FileType', 'text');

        fprintf('    Images and .dat files saved to: %s\n', outdir);
    end
end

%% ========================================================================
%  PRINT TABLE
% =========================================================================
fprintf('\n\n=========================================================================\n');
fprintf('  TABLE: Test 2 Streaming CT (1000x1000, 6 random subproblems)\n');
fprintf('  kmin=%d, kmax=%d\n', kmin, kmax);
fprintf('=========================================================================\n');
fprintf('  %-8s  %12s %12s   %12s %12s   %12s %16s\n', ...
    'sigma', 'MM-GKS 1st', 'LM-MM-GKS 1st', 'MM-GKS all', 'LM-MM-GKS all', 's-LM-MM-GKS', 's-LM-MM-GKS(tol)');
fprintf('  %s\n', repmat('-', 1, 95));
for ni = 1:n_noise
    fprintf('  %-8s  %12.4f %12.4f   %12.4f %12.4f   %12.4f %16.4f\n', ...
        sprintf('%.1f%%', noise_levels(ni)*100), RRE_table(ni,:));
end
fprintf('=========================================================================\n');

save(fullfile(outdir, 'test2_noiselevels_results.mat'), ...
    'RRE_table', 'noise_levels', 'kmin', 'kmax', 's', 'N', 'nblocks', 'blocksize');

fid = fopen(fullfile(outdir, 'test2_noiselevels_table.dat'), 'w');
fprintf(fid, 'sigma\tMMGKS_1st\tLMMGKS_1st\tMMGKS_all\tLMMGKS_all\tsLMMGKS\tsLMMGKS_tol\n');
for ni = 1:n_noise
    fprintf(fid, '%.1f%%\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        noise_levels(ni)*100, RRE_table(ni,:));
end
fclose(fid);

fprintf('\nResults saved to: %s\n', outdir);
fprintf('Done.\n');

%% ========================================================================
%  LOCAL FUNCTION
% =========================================================================
function rre = build_rre(si)
rre = [];
for k = 1:length(si)
    if isempty(si{k}), continue; end
    if isstruct(si{k}) && isfield(si{k}, 'Rerr')
        rr = si{k}.Rerr;
        rre = [rre; rr(:)];
    elseif isnumeric(si{k})
        rre = [rre; si{k}(:)];
    end
end
rre = rre(rre > 0);
end
