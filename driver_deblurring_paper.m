% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% =========================================================================
% Image deblurring experiment
%
% Problem:
%   500x500 Hubble telescope image, motion blur PSF (14x14 pixels),
%   0.1% additive Gaussian noise.
%
% Methods:
%   - MM-GKS (baseline, 25 iterations = memory capacity)
%   - LM-MM-GKS with four compression approaches: TSVD, RBD, SOC, SEC
%     for kmin = 5, 10, 15 with kmax = 25
% =========================================================================
clc
clear all
close all
rng(17, 'v4');

%% ---- Setup paths ----
directory = pwd;
addpath(directory)
addpath(fullfile(directory, 'utilities'))
addpath(fullfile(directory, 'HyBRrecycle'))
addpath(fullfile(directory, 'AIRToolsII-master'))
AIRToolsII_setup
addpath(fullfile(directory, 'IRTools'))
IRtools_setup

%% ---- Output directory ----
outdir = fullfile(pwd, 'results', 'Deblurring');
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ========================================================================
%  PROBLEM SETUP 
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
fprintf('  PSF size: %d x %d\n', size(PSF, 1), size(PSF, 2));

% Forward operator
A = Aclass(PSF, center, 'periodic', dd, dd);
b_true = A * x_true;

x_true = x_true(radius+1:end-radius, radius+1:end-radius);
b_true = b_true(radius+1:end-radius, radius+1:end-radius);
[nx, ny] = size(x_true);
n = nx * ny;
fprintf('  Image size: %d x %d (%d pixels)\n', nx, ny, n);
fprintf('  Operator size: %d x %d\n', n, n);

% Rebuild A 
A = Aclass(PSF, center, 'periodic', nx, ny);
b_true = b_true(:);

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
q            = 1;          
tol          = 1e-5;       
epsilon      = 0.001;      
kmax         = 25;         
kmin_values  = [5, 10, 15];
total_expand = 300;        

% MM-GKS baseline: 
mmgks_iter = kmax;

fprintf('\n  Parameters: q=%d, epsilon=%.0e, tol=%.0e, kmax=%d, total_expand=%d\n', ...
    q, epsilon, tol, kmax, total_expand);
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
    s = kmax - kmin;
    iter = total_expand / s;
    fprintf('\n====== kmin = %d, kmax = %d, s = %d, iter = %d (total = %d) ======\n', ...
        kmin, kmax, s, iter, iter*s);

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
        total_iters = info_rec.total_iter;
        sflag = info_rec.stop_flag;
        fprintf('    RRE = %.4f, HaarPSI = %.4f, iters = %d/%d (%.1f sec) [%s]\n', ...
            RRE, HP, total_iters, total_expand, elapsed, sflag);
        results(ki).(sprintf('RRE_%s', cname)) = RRE;
        results(ki).(sprintf('HP_%s', cname))  = HP;
        results(ki).(sprintf('time_%s', cname)) = elapsed;
        results(ki).(sprintf('stop_%s', cname)) = sflag;
        results(ki).(sprintf('iters_%s', cname)) = total_iters;
        results(ki).kmin = kmin;
        if kmin == 5
            recs.(cname) = x_rec;
        end
        if kmin == 10
            rre_curves.(cname) = build_full_rre(info_rec, si_rec);
        end
    end
end

%% ========================================================================
%  TABLE 1
% =========================================================================
fprintf('\n\n');
fprintf('=========================================================================\n');
fprintf('  TABLE 1: Image Deblurring — RRE and HaarPSI\n');
fprintf('  kmax = %d, sigma = %.1f%%, tol = %.0e, max %d expansion steps\n', kmax, sigma*100, tol, total_expand);
fprintf('=========================================================================\n');
fprintf('  MM-GKS (%d iters): RRE = %.4f, HaarPSI = %.4f\n\n', mmgks_iter, RRE_MMGKS, HP_MMGKS);

fprintf('  %4s  | %10s %10s %5s %5s | %10s %10s %5s %5s | %10s %10s %5s %5s | %10s %10s %5s %5s\n', ...
    'kmin', 'TSVD RRE', 'TSVD HP', 'iter', 'stop', 'RBD RRE', 'RBD HP', 'iter', 'stop', ...
    'SOC RRE', 'SOC HP', 'iter', 'stop', 'SEC RRE', 'SEC HP', 'iter', 'stop');
fprintf('  %s\n', repmat('-', 1, 160));
for ki = 1:length(kmin_values)
    r = results(ki);
    flags = {'TSVD', 'RBD', 'SOC', 'SEC'};
    short = cell(1,4);
    iters = zeros(1,4);
    for fi = 1:4
        sf = r.(sprintf('stop_%s', flags{fi}));
        iters(fi) = r.(sprintf('iters_%s', flags{fi}));
        if strcmp(sf, 'max_iter')
            short{fi} = 'max';
        else
            short{fi} = 'tol';
        end
    end
    fprintf('  %4d  | %10.4f %10.4f %5d %5s | %10.4f %10.4f %5d %5s | %10.4f %10.4f %5d %5s | %10.4f %10.4f %5d %5s\n', ...
        r.kmin, r.RRE_TSVD, r.HP_TSVD, iters(1), short{1}, ...
        r.RRE_RBD, r.HP_RBD, iters(2), short{2}, ...
        r.RRE_SOC, r.HP_SOC, iters(3), short{3}, ...
        r.RRE_SEC, r.HP_SEC, iters(4), short{4});
end
fprintf('=========================================================================\n');
fprintf('  iter: actual expansion steps used\n');
fprintf('  stop: ''tol'' = converged (rel_change <= %.0e), ''max'' = max iterations reached\n', tol);

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
clim_rec = [min(x_true(:)), max(x_true(:))];

methods_fig = {'MMGKS', 'TSVD', 'RBD', 'SEC', 'SOC'};
x_fig = {x_MMGKS, recs.TSVD, recs.RBD, recs.SEC, recs.SOC};
rre_fig = cellfun(@(x) norm(x(:)-x_true(:))/norm(x_true(:)), x_fig);

% Compute e_max across all methods
e_max = 0;
for j = 1:5
    e_max = max(e_max, max(abs(x_fig{j}(:) - x_true(:))));
end
fprintf('  e_max = %.4f\n', e_max);

figure('Position', [50 100 1100 400]);

% Row 1: Reconstructions
for j = 1:5
    subplot(2, 5, j);
    imagesc(reshape(x_fig{j}, nx, ny), clim_rec); axis image off; colormap gray;
    if j == 1
        title(sprintf('MM-GKS (25)\nRRE=%.4f', rre_fig(j)));
    else
        title(sprintf('%s\nRRE=%.4f', methods_fig{j}, rre_fig(j)));
    end
end

% Row 2: Error images (normalized by e_max)
for j = 1:5
    subplot(2, 5, 5+j);
    imagesc(reshape((x_fig{j}(:)-x_true(:)) / e_max, nx, ny), [-1, 0]); axis image off; colormap gray;
end

sgtitle('Figure 2: Reconstructions (top) and Error Images (bottom)', 'FontSize', 13);
exportgraphics(gcf, fullfile(outdir, 'fig2_reconstructions.pdf'), 'ContentType', 'vector');

%% ---- Individual images for paper ----
imgdir = fullfile(outdir, 'images');
if ~exist(imgdir, 'dir'), mkdir(imgdir); end

rec_names = {'deblurr_rec_MMGKS25', 'deblurr_rec_TSVD', 'deblurr_rec_RBD', ...
             'deblurr_rec_sparsity', 'deblurr_rec_solo'};
err_names = {'deblurr_err_MMGKS25', 'deblurr_err_SVD', 'deblurr_err_RBD', ...
             'deblurr_err_sparsity', 'deblurr_err_solo'};

for j = 1:5
    figure('Visible','off');
    imagesc(reshape(x_fig{j}, nx, ny), clim_rec); axis image off; colormap gray;
    set(gca,'Position',[0 0 1 1]);
    exportgraphics(gcf, fullfile(imgdir, [rec_names{j} '.jpg']), 'Resolution', 300); close

    figure('Visible','off');
    imagesc(reshape((x_fig{j}(:)-x_true(:)) / e_max, nx, ny), [-1, 0]); axis image off; colormap gray;
    set(gca,'Position',[0 0 1 1]);
    exportgraphics(gcf, fullfile(imgdir, [err_names{j} '.jpg']), 'Resolution', 300); close
end

figure('Visible','off');
imagesc(PSF); axis image off; colormap gray;
set(findobj(gca,'Type','image'), 'Interpolation', 'nearest');
set(gca,'Position',[0 0 1 1]);
exportgraphics(gcf, fullfile(imgdir, 'PSF.jpg'), 'Resolution', 300); close

fprintf('  Individual images saved to: %s\n', imgdir);

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
    final_rre = results(2).(sprintf('RRE_%s', cname));
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
    'kmin_values', 'kmax', 'sigma', 'total_expand', 'tol', 'mmgks_iter');

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
