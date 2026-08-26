% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% =========================================================================
% Image deblurring experiment (Paper Section 7.1).
%
% Problem:
%   500x500 Hubble telescope image, motion blur PSF (14x14 pixels),
%   0.1% additive Gaussian noise.
%
% Methods:
%   - MM-GKS (baseline, 25 iterations = memory capacity)
%   - LM-MM-GKS with four compression approaches: TSVD, RBD, SOC, SEC
%     for kmin = 5, 10, 15 with kmax = 25
%
% Outputs:
%   - Table 1: RRE and HaarPSI for each compression and kmin
%   - Figure 1: True image, PSF, blurred image
%   - Figure 2: Reconstructions and error images
%   - Figure 3: RRE convergence curves for all methods
%   - Results saved to results/ directory
%
% Usage:
%   >> addpath(pwd, 'RMMGKS2', 'AIRToolsII-master', 'IRTools')
%   >> AIRToolsII_setup; IRtools_setup
%   >> runme_deblurring_paper
% =========================================================================

clc
clear all
close all

%% ---- Setup paths ----
directory = pwd;
addpath(directory)
addpath(fullfile(directory, 'RMMGKS2'))
addpath(fullfile(directory, 'AIRToolsII-master'))
AIRToolsII_setup
addpath(fullfile(directory, 'IRTools'))
IRtools_setup

%% ---- Output directory ----
outdir = fullfile(pwd, 'results');
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ========================================================================
%  PROBLEM SETUP (Paper Section 7.1)
% =========================================================================
fprintf('Setting up image deblurring problem...\n');

% True image: Hubble Space Telescope
x_true = double(imread('HSTgray.jpg'));
dd = size(x_true, 1);
x_true = x_true / max(x_true(:)) * dd;

% PSF: motion blur (14x14 pixels)
PSF1 = fspecial('motion', 10, 45);
PSF1 = imresize(PSF1, 1.5);
PSF = PSF1 / sum(PSF1(:));
center = ceil(size(PSF) / 2);
radius = ceil(max(size(PSF)) / 2);

% Forward operator
A = Aclass(PSF, center, 'periodic', dd, dd);
b_true = A * x_true;

% Crop to remove PSF border effects
x_true = x_true(radius+1:end-radius, radius+1:end-radius);
b_true = b_true(radius+1:end-radius, radius+1:end-radius);
[nx, ny] = size(x_true);
n = nx * ny;
fprintf('  Image size: %d x %d (%d pixels)\n', nx, ny, n);
fprintf('  Operator size: %d x %d\n', n, n);

% Rebuild A for cropped size
A = Aclass(PSF, center, 'periodic', nx, ny);
b_true = b_true(:);

% Add noise (0.1% Gaussian)
rng(17, 'v4');
sigma = 0.001;
e_noise = randn(size(b_true));
e_noise = e_noise / norm(e_noise) * norm(b_true) * sigma;
b = b_true + e_noise;
noise_norm = norm(e_noise);
fprintf('  Noise level: %.1f%%\n', sigma * 100);

% Regularization operator: discrete gradient (paper eq 7.1)
LI = build_L(nx, ny, 1);
fprintf('  Regularization operator size: %d x %d\n', size(LI, 1), size(LI, 2));

%% ========================================================================
%  ALGORITHM PARAMETERS (Paper Section 7.1)
% =========================================================================
q       = 1;          % l_1 regularization
tol     = 0.01;       % inner convergence tolerance
epsilon = 0.001;      % smoothing parameter
kmax    = 25;         % max subspace dimension (memory capacity)
kmin_values = [5, 10, 15];  % compressed dimensions to test

% LM-MM-GKS iteration budget: iter x kiter = total max iterations
iter    = 25;         % outer expand-compress cycles
kiter   = 20;         % inner expansion steps per cycle (25 x 20 = 500 max)

% MM-GKS baseline: stopped at kmax = 25 iterations (memory limit)
mmgks_iter = kmax;

fprintf('\n  Parameters: q=%d, epsilon=%.0e, kmax=%d, max_iter=%d\n', ...
    q, epsilon, kmax, iter*kiter);
fprintf('  kmin values: [%s]\n', num2str(kmin_values));

%% ========================================================================
%  MM-GKS BASELINE (25 iterations, memory-limited)
% =========================================================================
fprintf('\n--- Running MM-GKS (%d iterations) ---\n', mmgks_iter);
rng(17, 'v4');
t0 = tic;
[x_MMGKS, ~, info_MMGKS] = ...
    MMGKS(A, b, LI, q, epsilon, mmgks_iter, tol, x_true(:));
time_MMGKS = toc(t0);

RRE_MMGKS = norm(x_MMGKS(:) - x_true(:)) / norm(x_true(:));
HP_MMGKS  = HaarPSI(x_true(:), x_MMGKS(:));
fprintf('  RRE = %.4f, HaarPSI = %.4f (%.1f sec) [%s]\n', ...
    RRE_MMGKS, HP_MMGKS, time_MMGKS, info_MMGKS.stop_flag);

% Extract per-iteration RRE from info
rre_MMGKS = info_MMGKS.Rerr;
rre_MMGKS = rre_MMGKS(rre_MMGKS > 0);

%% ========================================================================
%  LM-MM-GKS WITH FOUR COMPRESSION APPROACHES
% =========================================================================
compress_methods = {'TSVD', 'RBD', 'SOC', 'SEC'};
results = struct();
rre_curves = struct();

for ki = 1:length(kmin_values)
    kmin = kmin_values(ki);
    s = kmax - kmin;  % expansion steps per cycle (so kmin + s = kmax)
    fprintf('\n====== kmin = %d, kmax = %d (s = %d) ======\n', kmin, kmax, s);

    for ci = 1:length(compress_methods)
        cname = compress_methods{ci};
        fprintf('  Running LM-MM-GKS (%s, kmin=%d)...\n', cname, kmin);

        rng(17, 'v4');
        t1 = tic;
        [x_rec, ~, info_rec, si_rec] = LMMGKS( ...
            A, b, LI, q, epsilon, iter, s, kmin, tol, cname, x_true(:));
        elapsed = toc(t1);

        RRE = norm(x_rec(:) - x_true(:)) / norm(x_true(:));
        HP  = HaarPSI(x_true(:), x_rec(:));
        sflag = info_rec.stop_flag;
        fprintf('    RRE = %.4f, HaarPSI = %.4f (%.1f sec) [%s]\n', RRE, HP, elapsed, sflag);

        % Store results
        results(ki).(sprintf('RRE_%s', cname)) = RRE;
        results(ki).(sprintf('HP_%s', cname))  = HP;
        results(ki).(sprintf('time_%s', cname)) = elapsed;
        results(ki).(sprintf('stop_%s', cname)) = sflag;
        results(ki).kmin = kmin;

        % Store reconstruction for figures (use kmin=5)
        if kmin == 5
            recs.(cname) = x_rec;
        end

        % Store RRE convergence curve (use kmin=10 for convergence plot)
        if kmin == 10
            rre_curves.(cname) = build_full_rre(info_rec, si_rec);
        end
    end
end

%% ========================================================================
%  TABLE 1 (Paper Table 1)
% =========================================================================
fprintf('\n\n');
fprintf('=========================================================================\n');
fprintf('  TABLE 1: Image Deblurring — RRE and HaarPSI\n');
fprintf('  kmax = %d, sigma = %.1f%%, max %d iterations\n', kmax, sigma*100, iter*kiter);
fprintf('=========================================================================\n');
fprintf('  MM-GKS (%d iters): RRE = %.4f, HaarPSI = %.4f\n\n', mmgks_iter, RRE_MMGKS, HP_MMGKS);

fprintf('  %4s  | %10s %10s %5s | %10s %10s %5s | %10s %10s %5s | %10s %10s %5s\n', ...
    'kmin', 'TSVD RRE', 'TSVD HP', 'stop', 'RBD RRE', 'RBD HP', 'stop', ...
    'SOC RRE', 'SOC HP', 'stop', 'SEC RRE', 'SEC HP', 'stop');
fprintf('  %s\n', repmat('-', 1, 136));
for ki = 1:length(kmin_values)
    r = results(ki);
    % Short stop labels: 'tol' for converged, 'max' for max iterations
    flags = {'TSVD', 'RBD', 'SOC', 'SEC'};
    short = cell(1,4);
    for fi = 1:4
        sf = r.(sprintf('stop_%s', flags{fi}));
        if strcmp(sf, 'max_iter')
            short{fi} = 'max';
        else
            short{fi} = 'tol';
        end
    end
    fprintf('  %4d  | %10.4f %10.4f %5s | %10.4f %10.4f %5s | %10.4f %10.4f %5s | %10.4f %10.4f %5s\n', ...
        r.kmin, r.RRE_TSVD, r.HP_TSVD, short{1}, r.RRE_RBD, r.HP_RBD, short{2}, ...
        r.RRE_SOC, r.HP_SOC, short{3}, r.RRE_SEC, r.HP_SEC, short{4});
end
fprintf('=========================================================================\n');
fprintf('  stop: ''tol'' = converged (rel_change or residual), ''max'' = max iterations reached\n');

%% ========================================================================
%  FIGURE 1: True image, PSF, blurred image
% =========================================================================
figure('Position', [50 500 900 280]);

subplot(1,3,1);
imagesc(reshape(x_true, nx, ny)); axis image off; colormap gray;
title('(a) True image');

subplot(1,3,2);
imagesc(PSF); axis image off; colormap gray;
set(findobj(gca,'Type','image'), 'Interpolation', 'nearest');
title('(b) Motion blur PSF');

subplot(1,3,3);
imagesc(reshape(b, nx, ny)); axis image off; colormap gray;
title(sprintf('(c) Blurred + %.1f%% noise', sigma*100));

sgtitle('Figure 1: Image Deblurring Setup', 'FontSize', 13);
exportgraphics(gcf, fullfile(outdir, 'fig1_setup.pdf'), 'ContentType', 'vector');

%% ========================================================================
%  FIGURE 2: Reconstructions and error images (kmin=5)
% =========================================================================
rev_color = 100;
clim_rec = [min(x_true(:)), max(x_true(:))];

figure('Position', [50 100 1100 400]);

% Row 1: Reconstructions
methods_fig = {'MMGKS', 'TSVD', 'RBD', 'SEC', 'SOC'};
x_fig = {x_MMGKS, recs.TSVD, recs.RBD, recs.SEC, recs.SOC};
rre_fig = cellfun(@(x) norm(x(:)-x_true(:))/norm(x_true(:)), x_fig);

for j = 1:5
    subplot(2, 5, j);
    imagesc(reshape(x_fig{j}, nx, ny), clim_rec); axis image off; colormap gray;
    if j == 1
        title(sprintf('MM-GKS (25)\nRRE=%.4f', rre_fig(j)));
    else
        title(sprintf('%s\nRRE=%.4f', methods_fig{j}, rre_fig(j)));
    end
end

% Row 2: Error images
for j = 1:5
    subplot(2, 5, 5+j);
    imagesc(reshape(x_fig{j}(:)-x_true(:), nx, ny), [-rev_color, 0]); axis image off; colormap gray;
end

sgtitle('Figure 2: Reconstructions (top) and Error Images (bottom)', 'FontSize', 13);
exportgraphics(gcf, fullfile(outdir, 'fig2_reconstructions.pdf'), 'ContentType', 'vector');

%% ========================================================================
%  FIGURE 3: RRE convergence curves (kmin=10)
% =========================================================================
figure('Position', [50 50 700 450]);

colors = {'b', 'g', 'k', 'r'};
markers = {'s', '^', 'v', 'd'};

% Plot MM-GKS baseline
semilogy(1:length(rre_MMGKS), rre_MMGKS, 'm-o', 'LineWidth', 1.5, 'MarkerSize', 3, ...
    'DisplayName', sprintf('MM-GKS (%d iter, RRE=%.4f)', mmgks_iter, RRE_MMGKS));
hold on;

% Plot LM-MM-GKS for each compression
for ci = 1:length(compress_methods)
    cname = compress_methods{ci};
    rre = rre_curves.(cname);
    final_rre = results(2).(sprintf('RRE_%s', cname));  % kmin=10
    semilogy(1:length(rre), rre, [colors{ci} '-' markers{ci}], ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerIndices', 1:10:length(rre), ...
        'DisplayName', sprintf('LM-MM-GKS %s (RRE=%.4f)', cname, final_rre));
end
hold off;

xlabel('Iteration', 'FontSize', 12);
ylabel('RRE', 'FontSize', 12);
title(sprintf('RRE Convergence (k_{min}=%d, k_{max}=%d)', kmin_values(2), kmax), 'FontSize', 13);
legend('Location', 'northeast', 'FontSize', 9);
grid on;
exportgraphics(gcf, fullfile(outdir, 'fig3_convergence.pdf'), 'ContentType', 'vector');

%% ========================================================================
%  SAVE RESULTS
% =========================================================================
save(fullfile(outdir, 'deblurring_results.mat'), ...
    'results', 'RRE_MMGKS', 'HP_MMGKS', 'rre_MMGKS', 'rre_curves', ...
    'kmin_values', 'kmax', 'sigma', 'iter', 'kiter', 'mmgks_iter');

% Write table as .dat
fid = fopen(fullfile(outdir, 'table1.dat'), 'w');
fprintf(fid, 'kmin\tTSVD_RRE\tTSVD_HP\tRBD_RRE\tRBD_HP\tSOC_RRE\tSOC_HP\tSEC_RRE\tSEC_HP\n');
for ki = 1:length(kmin_values)
    r = results(ki);
    fprintf(fid, '%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        r.kmin, r.RRE_TSVD, r.HP_TSVD, r.RRE_RBD, r.HP_RBD, ...
        r.RRE_SOC, r.HP_SOC, r.RRE_SEC, r.HP_SEC);
end
fclose(fid);

fprintf('\nAll results saved to: %s\n', outdir);
fprintf('Done.\n');

%% ========================================================================
%  LOCAL FUNCTION
% =========================================================================
function rre_full = build_full_rre(info, si)
if ~isempty(si) && iscell(si)
    rre_full = [];
    for k = 1:length(si)
        if isstruct(si{k}) && isfield(si{k}, 'Rerr')
            inner_rre = si{k}.Rerr;
            rre_full = [rre_full; inner_rre(:)];
        elseif isnumeric(si{k}) && ~isempty(si{k})
            rre_full = [rre_full; si{k}(:)];
        end
    end
    rre_full = rre_full(rre_full > 0);
else
    if isfield(info, 'Rerr')
        rre_full = info.Rerr;
        rre_full = rre_full(rre_full > 0);
    else
        rre_full = [];
    end
end
end
