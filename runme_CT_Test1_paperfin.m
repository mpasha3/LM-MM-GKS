% Code maintainer: Mirjeta Pasha
% Contact: mpasha@vt.edu
% Copyright: Mirjeta Pasha, Eric de Sturler, Misha Kilmer
% Reference: "A Provably Convergent MM-GKS Variant for Large-Scale
%             Inverse Problems".
% =========================================================================
% Paper Section 7.2.1, Test 1: Streaming CT with 500x500 Shepp-Logan.
% Problem:
%   500x500 Shepp-Logan phantom, parallel tomography (IRTools).
% Methods (7 total):
%   1. HyBR 1st       — HyBR on first subproblem only
%   2. MM-GKS 1st     — MM-GKS on first subproblem only
%   3. HyBR all       — HyBR on full problem
%   4. MM-GKS all     — MM-GKS on full problem
%   5. LM-MM-GKS all  — LM-MM-GKS (TSVD) on full problem
%   6. HyBR-rec       — HyBR-recycle streaming across 3 subproblems (GCV)
%   7. s-LM-MM-GKS    — streaming LM-MM-GKS across 3 subproblems
% All methods: 200 expansion steps, GCV reg param selection.
% LM-MM-GKS: TSVD compression, kmin=10, kmax=40.
% HyBR-recycle: nInner=kmax=40, max_mm=kmin=10 (matching LM-MM-GKS).
% Figure 4: Reconstruction and error images (sigma = 0.1%)
% Figure 5: RRE convergence (s-LM-MM-GKS, HyBR-rec, s-LM-MM-GKS tol)
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
addpath(fullfile(directory, 'HyBR'))
addpath(fullfile(directory, 'AIRToolsII-master'))
AIRToolsII_setup
addpath(fullfile(directory, 'IRTools'))
IRtools_setup

%% ---- Output directories ----
outdir = fullfile(pwd, 'results', 'Test1');
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ========================================================================
%  PAPER PARAMETERS (Section 7.2.1)
% =========================================================================
N       = 500;
q       = 1;
epsilon = 0.001;
kmin    = 10;
kmax    = 40;
s       = kmax - kmin;  
maxit   = 200;          
tol     = 1e-8;         
tol_stream = 1e-3;      

iter_outer  = ceil(maxit / s);  
iter_per_sub = iter_outer;     
max_outer_tol = 20;            

noise_levels = [0.001, 0.005, 0.01];  

%% ========================================================================
%  GENERATE DATA 
% =========================================================================
fprintf('Generating CT data (Test 1, N=%d)...\n', N);

options = PRtomo('defaults');
options.angles = linspace(0, 44, 45);
[A1, b1_clean, x_true_vec] = PRtomo(N, options);
b1_clean = b1_clean(:);

options.angles = linspace(45, 89, 45);
[A2, b2_clean] = PRtomo(N, options);
b2_clean = b2_clean(:);

options.angles = linspace(90, 179, 45);
[A3, b3_clean] = PRtomo(N, options);
b3_clean = b3_clean(:);

x_true = x_true_vec(:);
nx = N; ny = N;

LI = build_L1(nx, ny, 1);

A_full = [A1; A2; A3];
b_clean_full = [b1_clean; b2_clean; b3_clean];
A_cell = {A1, A2, A3};
b_clean_cell = {b1_clean, b2_clean, b3_clean};

fprintf('  Image: %d x %d\n', N, N);
fprintf('  Rows per subproblem: %d\n', size(A1, 1));
fprintf('  Full system: %d x %d\n', size(A_full, 1), size(A_full, 2));
fprintf('  kmin=%d, kmax=%d, s=%d\n\n', kmin, kmax, s);

%% ---- Storage for Table 1 ----
n_noise = length(noise_levels);
RRE_table = zeros(n_noise, 7);

%% ========================================================================
%  LOOP OVER NOISE LEVELS
% =========================================================================
for ni = 1:n_noise
    sigma = noise_levels(ni);
    fprintf('=========================================================================\n');
    fprintf('  NOISE LEVEL: sigma = %.1f%%\n', sigma*100);
    fprintf('=========================================================================\n');

    % ---- Add noise ----
    rng(17, 'v4');
    b_cell = cell(1, 3);
    for p = 1:3
        ep = randn(size(b_clean_cell{p}));
        ep = ep / norm(ep) * norm(b_clean_cell{p}) * sigma;
        b_cell{p} = b_clean_cell{p} + ep;
    end
    b1 = b_cell{1}; b2 = b_cell{2}; b3 = b_cell{3};
    b_full = [b1; b2; b3];
    % ==================================================================
    %  METHOD 1: HyBR on first subproblem
    % ==================================================================
    fprintf('  1. HyBR 1st...\n');
    input_h = HyBRset('InSolv', 'Tikhonov', 'x_true', x_true, ...
        'Iter', maxit, 'RegPar', 'gcv', 'nLevel', sigma);
    rng(17, 'v4');
    [x_HyBR1, output_HyBR1] = HyBR(A1, b1, [], input_h);
    RRE_HyBR1 = norm(x_HyBR1(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_HyBR1);
    % ==================================================================
    %  METHOD 2: MM-GKS on first subproblem
    % ==================================================================
    fprintf('  2. MM-GKS 1st...\n');
    rng(17, 'v4');
    [x_MMGKS1, ~, info_MMGKS1] = MMGKS(A1, b1, LI, q, epsilon, maxit, tol, x_true);
    RRE_MMGKS1 = norm(x_MMGKS1(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_MMGKS1);
    % ==================================================================
    %  METHOD 3: HyBR on full problem
    % ==================================================================
    fprintf('  3. HyBR all...\n');
    input_h = HyBRset('InSolv', 'Tikhonov', 'x_true', x_true, ...
        'Iter', maxit, 'RegPar', 'gcv', 'nLevel', sigma);
    rng(17, 'v4');
    [x_HyBRall, output_HyBRall] = HyBR(A_full, b_full, [], input_h);
    RRE_HyBRall = norm(x_HyBRall(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_HyBRall);
    % ==================================================================
    %  METHOD 4: MM-GKS on full problem
    % ==================================================================
    fprintf('  4. MM-GKS all...\n');
    rng(17, 'v4');
    [x_MMGKSall, ~, info_MMGKSall] = MMGKS(A_full, b_full, LI, q, epsilon, maxit, tol, x_true);
    RRE_MMGKSall = norm(x_MMGKSall(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_MMGKSall);
    % ==================================================================
    %  METHOD 5: LM-MM-GKS on full problem (TSVD)
    % ==================================================================
    fprintf('  5. LM-MM-GKS all (TSVD, kmin=%d, kmax=%d)...\n', kmin, kmax);
    rng(17, 'v4');
    [x_LMMGKSall, ~, info_LMMGKSall, si_LMMGKSall] = ...
        LMMGKS(A_full, b_full, LI, q, epsilon, iter_outer, s, kmin, tol, 'TSVD', x_true);
    RRE_LMMGKSall = norm(x_LMMGKSall(:) - x_true) / norm(x_true);
    fprintf('    RRE = %.4f\n', RRE_LMMGKSall);
    % ==================================================================
    %  METHOD 6: HyBR-recycle (streaming, GCV)
    % ==================================================================
    fprintf('  6. HyBR-recycle (GCV, nInner=%d, max_mm=%d)...\n', kmax, kmin);
    trunc_options.nOuter = 6;
    trunc_options.nInner = kmax;
    trunc_options.max_mm = kmin;
    trunc_options.compress = 'SVD';
    trunc_mats = [];

    rng(17, 'v4');
    % Sub 1
    input_h = HyBRset('InSolv', 'Tikhonov', 'x_true', x_true, ...
        'Iter', trunc_options.nInner, 'Reorth', 'on', 'RegPar', 'gcv', 'nLevel', sigma);
    [~, tout_rec1, trunc_mats] = HyBRrecycle(A1, b1, [], input_h, trunc_options, trunc_mats);
    W = trunc_mats.W;
    trunc_mats.Y = []; trunc_mats.R = []; trunc_mats.x = []; trunc_mats.W = W;

    % Sub 2
    input_h = HyBRset('InSolv', 'Tikhonov', 'x_true', x_true, ...
        'Iter', trunc_options.nInner, 'RegPar', 'gcv', 'nLevel', sigma);
    [~, tout_rec2, trunc_mats] = HyBRrecycle(A2, b2, [], input_h, trunc_options, trunc_mats);
    W = trunc_mats.W;
    trunc_mats.Y = []; trunc_mats.R = []; trunc_mats.x = []; trunc_mats.W = W;

    % Sub 3
    [x_HyBRrec, tout_rec3, ~] = HyBRrecycle(A3, b3, [], input_h, trunc_options, trunc_mats);
    RRE_HyBRrec = norm(x_HyBRrec(:) - x_true) / norm(x_true);

    e1h = tout_rec1.Enrm; e1h = e1h(1:min(200, length(e1h)));
    e2h = tout_rec2.Enrm; e2h = e2h(1:min(200, length(e2h)));
    e3h = tout_rec3.Enrm; e3h = e3h(1:min(200, length(e3h)));
    err_HyBRrec_hist = [e1h; e2h; e3h];
    fprintf('    RRE = %.4f (%d iters)\n', RRE_HyBRrec, length(err_HyBRrec_hist));

    % ==================================================================
    %  METHOD 7: s-LM-MM-GKS
    % ==================================================================
    fprintf('  7. s-LM-MM-GKS (streaming, fixed iters)...\n');
    rng(17, 'v4');
    V_recycle = [];
    rre_sLMMGKS_hist = [];
    for p = 1:3
        [x_sub, ~, info_sub, si_sub] = ...
            LMMGKS(A_cell{p}, b_cell{p}, LI, q, epsilon, iter_per_sub, s, kmin, ...
            tol, 'TSVD', x_true, V_recycle);
        V_recycle = info_sub.oldV;
        rre_sLMMGKS_hist = [rre_sLMMGKS_hist; build_rre(si_sub)];
    end
    x_sLMMGKS = x_sub;
    RRE_sLMMGKS = norm(x_sLMMGKS(:) - x_true) / norm(x_true);
    rre_sLMMGKS_hist = rre_sLMMGKS_hist(:)';
    fprintf('    RRE = %.4f (%d iters)\n', RRE_sLMMGKS, length(rre_sLMMGKS_hist));

    % ---- Store Table 1 row ----
    RRE_table(ni, :) = [RRE_HyBR1, RRE_MMGKS1, RRE_HyBRall, RRE_MMGKSall, ...
                        RRE_LMMGKSall, RRE_HyBRrec, RRE_sLMMGKS];

    % ==================================================================
    %  FIGURES 
    % ==================================================================
    if abs(sigma - 0.001) < 1e-6
        fprintf('\n  Generating figures for sigma = 0.1%%...\n');

        outdir_fig = fullfile(outdir, 'figures');
        if ~exist(outdir_fig, 'dir'), mkdir(outdir_fig); end
        outdir_dat = fullfile(outdir, 'data');
        if ~exist(outdir_dat, 'dir'), mkdir(outdir_dat); end

        clim_rec = [0, max(x_true)];

        % ---- LM-MM-GKS 1st (needed for Figure 4 only) ----
        fprintf('    Running LM-MM-GKS 1st for Figure 4...\n');
        rng(17, 'v4');
        [x_LMMGKS1, ~, ~, ~] = ...
            LMMGKS(A1, b1, LI, q, epsilon, iter_per_sub, s, kmin, tol, 'TSVD', x_true);
        RRE_LMMGKS1 = norm(x_LMMGKS1(:) - x_true) / norm(x_true);
        fprintf('      RRE = %.4f\n', RRE_LMMGKS1);

        % ---- True image + sinograms ----
        n_rays = size(A1, 1) / 45;
        figure('Visible','off'); imagesc(reshape(x_true,N,N), clim_rec); axis image off; colormap gray;
        set(gca,'Position',[0 0 1 1]);
        exportgraphics(gcf, fullfile(outdir_fig, 'tomo_stream_3prob_true.jpg'), 'Resolution', 300); close

        sinos = {b1, b2, b3};
        for si = 1:3
            figure('Visible','off'); imagesc(reshape(sinos{si}, n_rays, 45)); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(outdir_fig, sprintf('tomo_stream_3prob_sino%d.jpg', si)), 'Resolution', 300); close
        end

        % ---- Figure 4: Reconstructions and errors ----
        row1_x = {x_HyBR1, x_MMGKS1, x_LMMGKS1, x_sLMMGKS};
        row1_names = {'rec_tomo_stream_3prob_HyBR1st', 'rec_tomo_stream_3prob_MMGKS_1st', ...
                      'rec_tomo_stream_3prob_RMMGKS_1st', 'rec_tomo_3prob_RMMGKS_3rd'};
        row2_names = {'err_tomo_stream_3prob_Hybr1st', 'err_tomo_stream_3prob_MMGKS1st', ...
                      'err_tomo_stream_3prob_RMMGKS1st', 'err_tomo_stream_3prob_RMMGKS_3rd'};
        for mi = 1:4
            figure('Visible','off'); imshow(reshape(row1_x{mi}, N, N), clim_rec, 'Border', 'tight');
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(outdir_fig, [row1_names{mi} '.jpg']), 'Resolution', 300); close

            figure('Visible','off'); imagesc(reshape(row1_x{mi}(:) - x_true, N, N), [-1, 0]); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(outdir_fig, [row2_names{mi} '.jpg']), 'Resolution', 300); close
        end

        row3_x = {x_HyBRall, x_HyBRrec, x_MMGKSall, x_LMMGKSall};
        row3_names = {'rec_tomo_stream_3prob_HyBR_all', 'rec_tomo_stream_3prob_HyBRrecycle_all', ...
                      'rec_tomo_stream_3prob_MMGKS_all', 'rec_tomo_stream_3prob_RMMGKS_all'};
        row4_names = {'err_tomo_stream_3prob_HyBR_all', 'err_tomo_stream_3prob_HyBR_recycle_all', ...
                      'err_tomo_stream_3prob_MMGKS_all', 'err_tomo_stream_3prob_RMMGKS_all'};
        for mi = 1:4
            figure('Visible','off'); imshow(reshape(row3_x{mi}, N, N), clim_rec, 'Border', 'tight');
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(outdir_fig, [row3_names{mi} '.jpg']), 'Resolution', 300); close

            figure('Visible','off'); imagesc(reshape(row3_x{mi}(:) - x_true, N, N), [-0.2, 0]); axis image off; colormap gray;
            set(gca,'Position',[0 0 1 1]);
            exportgraphics(gcf, fullfile(outdir_fig, [row4_names{mi} '.jpg']), 'Resolution', 300); close
        end
        fprintf('    Reconstruction images saved.\n');

        fprintf('    Running s-LM-MM-GKS (tol) for Figure 5...\n');
        rng(17, 'v4');
        V_recycle_tol = [];
        rre_sLMMGKS_tol_hist = [];
        for p = 1:3
            [x_sub_tol, ~, info_sub_tol, si_sub_tol] = ...
                LMMGKS(A_cell{p}, b_cell{p}, LI, q, epsilon, max_outer_tol, s, kmin, ...
                tol_stream, 'TSVD', x_true, V_recycle_tol);
            V_recycle_tol = info_sub_tol.oldV;
            rre_sLMMGKS_tol_hist = [rre_sLMMGKS_tol_hist; build_rre(si_sub_tol)];
        end
        rre_sLMMGKS_tol_hist = rre_sLMMGKS_tol_hist(:)';
        fprintf('      s-LM-MM-GKS (tol): RRE = %.4f (%d iters)\n', rre_sLMMGKS_tol_hist(end), length(rre_sLMMGKS_tol_hist));

        rre_s = rre_sLMMGKS_hist(:);
        T = table((1:length(rre_s))', rre_s, 'VariableNames', {'Iteration', 'RRE'});
        writetable(T, fullfile(outdir_dat, 'tomo_sLMMGKS.dat'), 'Delimiter', '\t', 'FileType', 'text');

        err_h = err_HyBRrec_hist(:);
        T = table((1:length(err_h))', err_h, 'VariableNames', {'Iteration', 'RRE'});
        writetable(T, fullfile(outdir_dat, 'tomo_HyBRrec.dat'), 'Delimiter', '\t', 'FileType', 'text');

        rre_t = rre_sLMMGKS_tol_hist(:);
        T = table((1:length(rre_t))', rre_t, 'VariableNames', {'Iteration', 'RRE'});
        writetable(T, fullfile(outdir_dat, 'tomo_sLMMGKS_tol.dat'), 'Delimiter', '\t', 'FileType', 'text');
        fprintf('    .dat files saved.\n');

        mk1 = max(1, round(length(rre_sLMMGKS_hist)/15));
        mk2 = max(1, round(length(err_HyBRrec_hist)/15));
        mk3 = max(1, round(length(rre_sLMMGKS_tol_hist)/15));

        fig5 = figure('Name', 'Paper Figure 5', 'NumberTitle', 'off', 'Position', [100 100 900 500]);
        semilogy(1:length(rre_sLMMGKS_hist), rre_sLMMGKS_hist, 'b-*', ...
            'LineWidth', 1.5, 'MarkerSize', 4, ...
            'MarkerIndices', 1:mk1:length(rre_sLMMGKS_hist)), hold on
        semilogy(1:length(err_HyBRrec_hist), err_HyBRrec_hist, 'r-o', ...
            'LineWidth', 1.5, 'MarkerSize', 4, ...
            'MarkerIndices', 1:mk2:length(err_HyBRrec_hist))
        semilogy(1:length(rre_sLMMGKS_tol_hist), rre_sLMMGKS_tol_hist, 'm-^', ...
            'LineWidth', 1.5, 'MarkerSize', 4, ...
            'MarkerIndices', 1:mk3:length(rre_sLMMGKS_tol_hist))
        xlabel('Iteration', 'FontSize', 13)
        ylabel('RRE', 'FontSize', 13)
        legend('s-LM-MM-GKS', 'HyBR-recycle (GCV)', ...
            sprintf('s-LM-MM-GKS (tol = %g)', tol_stream), ...
            'Location', 'best', 'FontSize', 10)
        grid on, hold off
        saveas(fig5, fullfile(outdir_fig, 'fig5_RRE_convergence.jpg'));
        exportgraphics(fig5, fullfile(outdir_fig, 'fig5_RRE_convergence.pdf'), 'ContentType', 'vector');
        fprintf('    Figure 5 saved.\n');
    end
end

%% ========================================================================
%  PRINT TABLE 1
% =========================================================================
fprintf('\n\n=========================================================================\n');
fprintf('  TABLE 1: Streaming CT Test 1 (%dx%d, 3 subproblems)\n', N, N);
fprintf('  kmin=%d, kmax=%d, %d expansion steps\n', kmin, kmax, maxit);
fprintf('=========================================================================\n');
fprintf('           | Single subproblem |   Full problem              | Streaming          \n');
fprintf('  sigma    | HyBR 1st  MMGKS 1st | HyBR all  MMGKS all  LMMGKS all | HyBR-rec  s-LMMGKS\n');
fprintf('  %s\n', repmat('-', 1, 90));
for ni = 1:n_noise
    fprintf('  %-7s  | %7.4f   %7.4f   | %7.4f   %7.4f    %7.4f    | %7.4f   %7.4f\n', ...
        sprintf('%.1f%%', noise_levels(ni)*100), RRE_table(ni,:));
end
fprintf('=========================================================================\n');

%% ========================================================================
%  SAVE RESULTS
% =========================================================================
save(fullfile(outdir, 'test1_table1_results.mat'), ...
    'RRE_table', 'noise_levels', 'kmin', 'kmax', 's', 'maxit', 'N');

fid = fopen(fullfile(outdir, 'test1_table1.dat'), 'w');
fprintf(fid, 'sigma\tHyBR_1st\tMMGKS_1st\tHyBR_all\tMMGKS_all\tLMMGKS_all\tHyBR_rec\tsLMMGKS\n');
for ni = 1:n_noise
    fprintf(fid, '%.1f%%\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        noise_levels(ni)*100, RRE_table(ni,:));
end
fclose(fid);

fprintf('\nAll results saved to: %s\n', outdir);
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
