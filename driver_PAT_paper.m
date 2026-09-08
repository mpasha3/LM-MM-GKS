% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
%
% =========================================================================
% Paper Section 7.2: Dynamic Photoacoustic Tomography (PAT).
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
outdir = fullfile(pwd, 'results', 'PAT');
if ~exist(outdir, 'dir'), mkdir(outdir); end
outdir_fig = fullfile(outdir, 'figures');
if ~exist(outdir_fig, 'dir'), mkdir(outdir_fig); end
outdir_dat = fullfile(outdir, 'data');
if ~exist(outdir_dat, 'dir'), mkdir(outdir_dat); end

%% ========================================================================
%  PAPER PARAMETERS (Section 7.2)
% =========================================================================
nt      = 50;       
nx      = 256;     
q       = 1;        
epsilon = 0.001;    
kmin    = 5;        
kmax    = 15;       
s       = kmax - kmin;  
total_iter = 500;   
iter    = total_iter / s;  
tol     = 1e-8;     
mmgks_iter = kmax; 

noise_levels = [0.001, 0.005, 0.01, 0.05];  

%% ========================================================================
%  GENERATE PAT DATA
% =========================================================================
fprintf('Generating PAT data (nt=%d, nx=%d)...\n', nt, nx);
[A, ~, Amat, ~, nx, ny, nt, ~, ~, x_true, b_clean, B_clean] = ...
    generate_PAT(nx, nx, nt, 1, 0.01);

x_true = x_true(:);

fprintf('  Images: %d x %d x %d\n', nx, ny, nt);
fprintf('  Measurements per time step: %d\n', size(Amat{1}, 1));
fprintf('  Total measurements: %d\n', length(b_clean));
fprintf('  Total unknowns: %d\n', nx*ny*nt);

%% ---- Build regularization operator ----
order = 1;
[LI, ~] = build_L_R1a(nx, ny, nt, order);
fprintf('  Regularization operator Psi: %d x %d\n', size(LI));

%% ---- Storage for Table 3 ----
n_noise = length(noise_levels);
RRE_table  = zeros(n_noise, 3); 
SSIM_table = zeros(n_noise, 3);
PSNR_table = zeros(n_noise, 3);

%% ========================================================================
%  LOOP OVER NOISE LEVELS
% =========================================================================
for ni = 1:n_noise
    sigma = noise_levels(ni);
    fprintf('\n=========================================================================\n');
    fprintf('  NOISE LEVEL: sigma = %.1f%%\n', sigma*100);
    fprintf('=========================================================================\n');

    % ---- Add noise to clean data ----
    rng(17, 'v4');
    B_noisy = zeros(size(B_clean));
    for i = 1:nt
        ei = randn(size(B_clean(:,i)));
        ei = ei / norm(ei) * norm(B_clean(:,i)) * sigma;
        B_noisy(:,i) = B_clean(:,i) + ei;
    end
    b = B_noisy(:);
    noise_norm = norm(b - b_clean);

    % ==================================================================
    %  METHOD 1: MM-GKS
    % ==================================================================
    fprintf('  1. MM-GKS (%d iterations, GCV, eps=%.0e)...\n', mmgks_iter, epsilon);
    rng(17, 'v4');
    [x_MMGKS, ~, info_MMGKS] = MMGKS(A, b, LI, q, epsilon, mmgks_iter, tol, x_true);

    saveX_MMGKS = info_MMGKS.saveX;
    err_MMGKS = zeros(1, mmgks_iter);
    for i = 1:mmgks_iter
        err_MMGKS(i) = norm(saveX_MMGKS(:,i) - x_true) / norm(x_true);
    end
    RRE_MMGKS  = err_MMGKS(end);
    SSIM_MMGKS = ssim(x_MMGKS(:), x_true);
    PSNR_MMGKS = psnr(x_MMGKS(:), x_true);
    fprintf('    RRE = %.4f, SSIM = %.4f, PSNR = %.2f\n', RRE_MMGKS, SSIM_MMGKS, PSNR_MMGKS);

    % ==================================================================
    %  METHOD 2: MM-GKS_res 
    % ==================================================================
    fprintf('  2. MM-GKS_res (lplq_res, DP adaptive, restart=%d, eps=1)...\n', kmax);
    opts = lplq_res('defaults');
    opts.p            = 2;
    opts.q            = q;
    opts.L            = LI;
    opts.epsilon      = 1;          
    opts.tol          = 1e-8;
    opts.maxIt        = total_iter;
    opts.restart      = kmax;
    opts.choiceRule    = 'DP';
    opts.majorantType = 'adaptive';
    opts.noiseNorm    = noise_norm;
    opts.xTrue        = x_true;

    rng(17, 'v4');
    [x_MMGKSres, outInfo_MMGKSres] = lplq_res(A, b, opts);

    niter_MMGKSres = outInfo_MMGKSres.iter;
    Rerr_MMGKSres  = outInfo_MMGKSres.RRE(1:niter_MMGKSres+1)';
    RRE_MMGKSres   = Rerr_MMGKSres(end);
    SSIM_MMGKSres  = ssim(x_MMGKSres(:), x_true);
    PSNR_MMGKSres  = psnr(x_MMGKSres(:), x_true);
    fprintf('    RRE = %.4f, SSIM = %.4f, PSNR = %.2f (%d iters)\n', ...
        RRE_MMGKSres, SSIM_MMGKSres, PSNR_MMGKSres, niter_MMGKSres);

    % ==================================================================
    %  METHOD 3: LM-MM-GKS (Algorithm 3.3, TSVD compression)
    % ==================================================================
    fprintf('  3. LM-MM-GKS (TSVD, kmin=%d, kmax=%d, %d iters, GCV, eps=%.0e)...\n', ...
        kmin, kmax, total_iter, epsilon);
    rng(17, 'v4');
    [x_LMMGKS, ~, info_LMMGKS, saveinfo_inner_LMMGKS] = ...
        LMMGKS(A, b, LI, q, epsilon, iter, s, kmin, tol, 'TSVD', x_true);

    saveX_LMMGKS = info_LMMGKS.saveX;
    niter_LMMGKS = size(saveX_LMMGKS, 2);
    err_LMMGKS = zeros(1, niter_LMMGKS);
    for i = 1:niter_LMMGKS
        err_LMMGKS(i) = norm(saveX_LMMGKS(:,i) - x_true) / norm(x_true);
    end

    Rerr_LMMGKS = [];
    for i = 1:length(saveinfo_inner_LMMGKS)
        ri = saveinfo_inner_LMMGKS{i};
        if isempty(ri), continue; end
        if isstruct(ri) && isfield(ri, 'Rerr')
            ri = ri.Rerr;
        end
        if length(ri) > 1
            ri = ri(2:end); 
        end
        if isempty(ri), continue; end
        Rerr_LMMGKS = [Rerr_LMMGKS; ri(:)];
    end
    Rerr_LMMGKS = Rerr_LMMGKS(Rerr_LMMGKS > 0);  
    Rerr_LMMGKS = Rerr_LMMGKS(:)';  

    RRE_LMMGKS  = err_LMMGKS(end);
    SSIM_LMMGKS = ssim(x_LMMGKS(:), x_true);
    PSNR_LMMGKS = psnr(x_LMMGKS(:), x_true);
    fprintf('    RRE = %.4f, SSIM = %.4f, PSNR = %.2f (%d inner iters)\n', ...
        RRE_LMMGKS, SSIM_LMMGKS, PSNR_LMMGKS, length(Rerr_LMMGKS));

    % ---- Store for Table 3 ----
    RRE_table(ni, :)  = [RRE_MMGKS,  RRE_MMGKSres,  RRE_LMMGKS];
    SSIM_table(ni, :) = [SSIM_MMGKS, SSIM_MMGKSres, SSIM_LMMGKS];
    PSNR_table(ni, :) = [PSNR_MMGKS, PSNR_MMGKSres, PSNR_LMMGKS];

    % ==================================================================
    %  FIGURES AND DATA FILES
    % ==================================================================
    if abs(sigma - 0.01) < 1e-6
        fprintf('\n  Generating figures for sigma = 1%%...\n');

        x_true_3d     = reshape(x_true, nx, nx, []);
        x_MMGKS_3d    = reshape(x_MMGKS(:), nx, nx, []);
        x_MMGKSres_3d = reshape(x_MMGKSres(:), nx, nx, []);
        x_LMMGKS_3d   = reshape(x_LMMGKS(:), nx, nx, []);

        img_err_MMGKS    = reshape(x_MMGKS(:)    - x_true, nx, nx, []);
        img_err_MMGKSres = reshape(x_MMGKSres(:) - x_true, nx, nx, []);
        img_err_LMMGKS   = reshape(x_LMMGKS(:)   - x_true, nx, nx, []);

        % Compute e_max across all 3 methods and all time steps
        e_max = max([max(abs(img_err_MMGKS(:))), ...
                     max(abs(img_err_MMGKSres(:))), ...
                     max(abs(img_err_LMMGKS(:)))]);
        fprintf('    e_max = %.4f\n', e_max);

        time_indices = [1, 10, 20, 30, 40, 50];

        for idx = 1:length(time_indices)
            t = time_indices(idx);
            if t > size(x_true_3d, 3), continue; end

            % ---- True images ----
            fig = figure('Visible', 'off');
            imagesc(x_true_3d(:,:,t)); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('PAT_true_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % ---- Sinograms ------
            fig = figure('Visible', 'off');
            sino_t = B_noisy(:, t);
            n_angles = 49;
            n_radii = length(sino_t) / n_angles;
            imagesc(reshape(sino_t, n_radii, n_angles)); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('PAT_sino_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % ---- Reconstructions ----
            % MM-GKS
            fig = figure('Visible', 'off');
            imagesc(x_MMGKS_3d(:,:,t)); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('rec_MMGKS_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % MM-GKS_res
            fig = figure('Visible', 'off');
            imagesc(x_MMGKSres_3d(:,:,t)); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('rec_res_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % LM-MM-GKS
            fig = figure('Visible', 'off');
            imagesc(x_LMMGKS_3d(:,:,t)); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('rec_rec_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % ---- Error images (normalized by e_max, displayed on [-1, 0]) ----
            % Row 1: MM-GKS at 15 iterations
            fig = figure('Visible', 'off');
            imagesc(img_err_MMGKS(:,:,t) / e_max, [-1, 0]); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('err_MMGKS_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % Row 2: MM-GKS_res
            fig = figure('Visible', 'off');
            imagesc(img_err_MMGKSres(:,:,t) / e_max, [-1, 0]); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('err_res_%d.jpg', t)), 'Resolution', 300);
            close(fig)

            % Row 3: LM-MM-GKS
            fig = figure('Visible', 'off');
            imagesc(img_err_LMMGKS(:,:,t) / e_max, [-1, 0]); axis image off; colormap gray;
            set(gca, 'Position', [0 0 1 1]);
            exportgraphics(fig, fullfile(outdir_fig, sprintf('err_rec_%d.jpg', t)), 'Resolution', 300);
            close(fig)
        end
        fprintf('    Images saved.\n');

        % ---- .dat files for tikz/pgfplots (paper Figure 8) ----
        T = table((1:length(Rerr_LMMGKS))', Rerr_LMMGKS', ...
            'VariableNames', {'Iteration', 'RRE'});
        writetable(T, fullfile(outdir_dat, 'PAT_LMMGKS.dat'), ...
            'Delimiter', '\t', 'FileType', 'text');

        T = table((1:mmgks_iter)', err_MMGKS', ...
            'VariableNames', {'Iteration', 'RRE'});
        writetable(T, fullfile(outdir_dat, 'PAT_MMGKS.dat'), ...
            'Delimiter', '\t', 'FileType', 'text');

        T = table((0:niter_MMGKSres)', Rerr_MMGKSres', ...
            'VariableNames', {'Iteration', 'RRE'});
        writetable(T, fullfile(outdir_dat, 'PAT_MMGKSres.dat'), ...
            'Delimiter', '\t', 'FileType', 'text');
        fprintf('    .dat files saved.\n');

        % ---- RRE convergence plot (paper Figure 8) ----
        fprintf('    Rerr_LMMGKS: %d elements, class=%s\n', length(Rerr_LMMGKS), class(Rerr_LMMGKS));
        fprintf('    err_MMGKS: %d elements, class=%s\n', length(err_MMGKS), class(err_MMGKS));
        fprintf('    Rerr_MMGKSres: %d elements, class=%s\n', length(Rerr_MMGKSres), class(Rerr_MMGKSres));
        fig1 = figure('Name', 'PAT RRE convergence', 'NumberTitle', 'off', ...
            'Position', [100 100 900 550]);
        semilogy(1:length(Rerr_LMMGKS), Rerr_LMMGKS, 'b-', 'LineWidth', 1.5), hold on
        semilogy(1:mmgks_iter, err_MMGKS, 'r--', 'LineWidth', 1.5)
        semilogy(0:niter_MMGKSres, Rerr_MMGKSres, 'g-.', 'LineWidth', 1.5)
        xlabel('Iteration', 'FontSize', 13)
        ylabel('RRE', 'FontSize', 13)
        legend('LM-MM-GKS', sprintf('MM-GKS_{%d}', mmgks_iter), 'MM-GKS_{res}', ...
            'Location', 'best', 'FontSize', 11)
        grid on, hold off
        saveas(fig1, fullfile(outdir_fig, 'err_conv_PAT.jpg'));
        exportgraphics(fig1, fullfile(outdir_fig, 'err_conv_PAT.pdf'), 'ContentType', 'vector');
        fprintf('    RRE convergence plot saved.\n');
    end
end

%% ========================================================================
%  TABLE 3 (paper format)
% =========================================================================
fprintf('\n\n=========================================================================\n');
fprintf('  TABLE 3: Dynamic PAT (%dx%d, nt=%d)\n', nx, nx, nt);
fprintf('  kmin=%d, kmax=%d, 500 iterations (MM-GKS stopped at %d)\n', kmin, kmax, mmgks_iter);
fprintf('  MM-GKS & LM-MM-GKS: GCV, eps=1e-3. MM-GKS_res: DP, eps=1.\n');
fprintf('=========================================================================\n');
fprintf('           |     MM-GKS_{15}       |     MM-GKS_{res}      |      LM-MM-GKS       \n');
fprintf('  sigma    |   RRE    SSIM   PSNR  |   RRE    SSIM   PSNR  |   RRE    SSIM   PSNR \n');
fprintf('  %s\n', repmat('-', 1, 85));
for ni = 1:n_noise
    fprintf('  %-7s  | %6.4f  %6.4f  %5.2f  | %6.4f  %6.4f  %5.2f  | %6.4f  %6.4f  %5.2f\n', ...
        sprintf('%.1f%%', noise_levels(ni)*100), ...
        RRE_table(ni,1), SSIM_table(ni,1), PSNR_table(ni,1), ...
        RRE_table(ni,2), SSIM_table(ni,2), PSNR_table(ni,2), ...
        RRE_table(ni,3), SSIM_table(ni,3), PSNR_table(ni,3));
end
fprintf('=========================================================================\n');

%% ========================================================================
%  SAVE RESULTS
% =========================================================================
save(fullfile(outdir, 'PAT_table3_results.mat'), ...
    'RRE_table', 'SSIM_table', 'PSNR_table', 'noise_levels', ...
    'kmin', 'kmax', 'total_iter', 'nx', 'nt', 'epsilon');

fid = fopen(fullfile(outdir, 'PAT_table3.dat'), 'w');
fprintf(fid, 'sigma\tRRE_MMGKS\tSSIM_MMGKS\tPSNR_MMGKS\tRRE_MMGKSres\tSSIM_MMGKSres\tPSNR_MMGKSres\tRRE_LMMGKS\tSSIM_LMMGKS\tPSNR_LMMGKS\n');
for ni = 1:n_noise
    fprintf(fid, '%.3f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f\t%.2f\n', ...
        noise_levels(ni), ...
        RRE_table(ni,1), SSIM_table(ni,1), PSNR_table(ni,1), ...
        RRE_table(ni,2), SSIM_table(ni,2), PSNR_table(ni,2), ...
        RRE_table(ni,3), SSIM_table(ni,3), PSNR_table(ni,3));
end
fclose(fid);

fprintf('\nAll results saved to: %s\n', outdir);
fprintf('Done.\n');
