function qc_corr(WorkDir, s, SPMRoot)
addpath(SPMRoot);
spm('defaults', 'fmri'); spm_jobman('initcfg');
q = fullfile(WorkDir, 'QC', s); if ~exist(q, 'dir'); mkdir(q); end
m = dir(fullfile(WorkDir, 'RealignParameter', s, 'mean*.nii'));
y = dir(fullfile(WorkDir, 'T1ImgNewSegment', s, 'y_*.nii'));
copyfile(fullfile(WorkDir, 'RealignParameter', s, m(1).name), fullfile(q, m(1).name));
job = struct();
job.spm.spatial.normalise.write.subj.def = {fullfile(WorkDir, 'T1ImgNewSegment', s, y(1).name)};
job.spm.spatial.normalise.write.subj.resample = {fullfile(q, m(1).name)};
job.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72; 90 90 108];
job.spm.spatial.normalise.write.woptions.vox = [3 3 3];
job.spm.spatial.normalise.write.woptions.interp = 1;
job.spm.spatial.normalise.write.woptions.prefix = 'w';
spm_jobman('run', {job});
f = dir(fullfile(WorkDir, 'FunImgARCFW', s, 'w*.nii'));
V = spm_vol(fullfile(WorkDir, 'FunImgARCFW', s, f(1).name));
Y = spm_read_vols(V);
M = double(std(Y, 0, 4) > 0);
Vo = V(1); Vo.fname = fullfile(q, 'brainmask_w.nii'); Vo.dt = [spm_type('uint8') 0]; Vo.n = [1 1];
spm_write_vol(Vo, M);
fprintf('P6_QC_OK %s\n', s);
end
