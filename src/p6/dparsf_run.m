function dparsf_run(WorkDir, SliceTiming, SubjectList, DPABIRoot, SPMRoot)
if exist('OCTAVE_VERSION', 'builtin')   % Octave Forge packages providing MATLAB toolbox functions DPARSF uses
  pkg load image; pkg load signal; pkg load statistics;
  save_default_options('-v7');           % DPARSF save() calls must write MATLAB-format .mat files
end
addpath(genpath(DPABIRoot));
rmpath(genpath(fullfile(DPABIRoot, 'RedistributedToolboxes', 'palm-alpha115')));  % ships stale Octave MEX files
addpath(SPMRoot);  % added last so SPM12's own (Octave-compiled) functions take precedence
spm('defaults', 'fmri');
spm_jobman('initcfg');

Done1a = fullfile(WorkDir, 'P6_stage1a.done');
if ~exist(Done1a, 'file')
Cfg = base_cfg(DPABIRoot, WorkDir, SubjectList);
Cfg.StartingDirName = 'FunImg';
Cfg.IsSliceTiming = SliceTiming;
Cfg.SliceTiming.SliceNumber = 0;          % read SliceTiming from the JSON sidecar
Cfg.IsRemoveFirstTimePoints = 1;
Cfg.RemoveFirstTimePoints = 10;
Cfg.IsSegment = 2;                        % New Segment
Cfg.IsCovremove = 0;
Cfg.IsFilter = 0;
save(fullfile(WorkDir, 'Cfg_stage1a.mat'), 'Cfg', '-v7');
[Error] = DPARSFA_run(Cfg, WorkDir, '', 0); if ~isempty(Error); disp(Error); end
fclose(fopen(Done1a, 'w'));
end
fprintf('P6_STAGE1A_DONE\n');

Done1b = fullfile(WorkDir, 'P6_stage1b.done');
if ~exist(Done1b, 'file')
TemplatePath = fullfile(DPABIRoot, 'Templates');
masks = {'BrainMask_05_91x109x91', 'CsfMask_07_91x109x91', 'WhiteMask_09_91x109x91', 'GreyMask_02_91x109x91'};
OutMaskDir = fullfile(WorkDir, 'Masks', 'WarpedMasks');
if ~exist(OutMaskDir, 'dir'); mkdir(OutMaskDir); end
for i = 1:numel(SubjectList)
  s = SubjectList{i};
  iy = dir(fullfile(WorkDir, 'T1ImgNewSegment', s, 'iy_*.nii'));
  m = dir(fullfile(WorkDir, 'RealignParameter', s, 'mean*.nii'));
  RefFile = fullfile(WorkDir, 'RealignParameter', s, m(1).name);
  [~, RefVox] = y_ReadRPI(RefFile, 1);
  tmpdir = fullfile(OutMaskDir, ['tmp_' s]); mkdir(tmpdir);
  job = struct();
  job.spm.util.defs.comp{1}.def = {fullfile(WorkDir, 'T1ImgNewSegment', s, iy(1).name)};
  job.spm.util.defs.out{1}.pull.fnames = cellfun(@(x) fullfile(TemplatePath, [x '.img']), masks, 'UniformOutput', false)';
  job.spm.util.defs.out{1}.pull.savedir.saveusr = {tmpdir};
  job.spm.util.defs.out{1}.pull.interp = 0;
  job.spm.util.defs.out{1}.pull.mask = 0;
  job.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
  job.spm.util.defs.out{1}.pull.prefix = 'w';
  spm_jobman('run', {job});
  for k = 1:numel(masks)
    w = dir(fullfile(tmpdir, ['w' masks{k} '.*']));
    w = w(~cellfun(@isempty, regexp({w.name}, '\.(nii|img)$')));
    y_Reslice(fullfile(tmpdir, w(1).name), fullfile(OutMaskDir, [s '_' masks{k} '.nii']), RefVox, 0, RefFile);
  end
  rmdir(tmpdir, 's');
end
fclose(fopen(Done1b, 'w'));
end
fprintf('P6_STAGE1B_DONE\n');

Done1c = fullfile(WorkDir, 'P6_stage1c.done');
Cfg = base_cfg(DPABIRoot, WorkDir, SubjectList);
Cfg.StartingDirName = 'FunImgAR';
if ~SliceTiming; Cfg.StartingDirName = 'FunImgR'; end
Cfg.IsRemoveFirstTimePoints = 0;
Cfg.RemoveFirstTimePoints = 0;            % DPARSF trims (in place) whenever this is > 0, regardless of the flag
Cfg.IsSliceTiming = 0;
Cfg.IsRealign = 0;
Cfg.IsBet = 0;
Cfg.IsNeedT1CoregisterToFun = 0;
Cfg.IsSegment = 0;
Cfg.IsCovremove = 1;                      % Covremove settings unchanged from the template
Cfg.IsFilter = 1;
Cfg.Filter.Timing = 'BeforeNormalize';
if ~exist(Done1c, 'file')
save(fullfile(WorkDir, 'Cfg_stage1c.mat'), 'Cfg', '-v7');
[Error] = DPARSFA_run(Cfg, WorkDir, '', 0); if ~isempty(Error); disp(Error); end
fclose(fopen(Done1c, 'w'));
end
fprintf('P6_STAGE1C_DONE\n');

InDir = fullfile(WorkDir, [Cfg.StartingDirName 'CF']);
OutRoot = fullfile(WorkDir, 'Results', 'ROISignals_FunImgARCFW');
if ~exist(OutRoot, 'dir'); mkdir(OutRoot); end
AAL = fullfile(DPABIRoot, 'Templates', 'aal.nii');
AALr = '';
for i = 1:numel(SubjectList)
  s = SubjectList{i};
  f = dir(fullfile(InDir, s, '*.nii'));
  y = dir(fullfile(WorkDir, 'T1ImgNewSegment', s, 'y_*.nii'));
  if isempty(f) || isempty(y); fprintf('P6_MISSING %s\n', s); continue; end
  src = fullfile(InDir, s, f(1).name);
  OutDir = fullfile(WorkDir, 'FunImgARCFW', s);
  wout = fullfile(OutDir, ['w' f(1).name]);
  if ~exist(wout, 'file')
  job = struct();
  job.spm.spatial.normalise.write.subj.def = {fullfile(WorkDir, 'T1ImgNewSegment', s, y(1).name)};
  job.spm.spatial.normalise.write.subj.resample = cellstr(spm_select('expand', src));
  job.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72; 90 90 108];
  job.spm.spatial.normalise.write.woptions.vox = [3 3 3];
  job.spm.spatial.normalise.write.woptions.interp = 1;
  job.spm.spatial.normalise.write.woptions.prefix = 'w';
  spm_jobman('run', {job});
  if ~exist(OutDir, 'dir'); mkdir(OutDir); end
  movefile(fullfile(InDir, s, ['w' f(1).name]), wout);
  end
  if isempty(AALr)   % reslice the 1 mm AAL to the 3 mm data grid once (nearest neighbour), as DPARSF does
    AALr = fullfile(WorkDir, 'Masks', 'aal_3mm_resliced.nii');
    Ref3D = fullfile(WorkDir, 'Masks', 'ref_3mm_grid.nii');
    [Data, Head] = y_Read(wout); y_Write(Data(:, :, :, 1), Head, Ref3D); clear Data;
    [~, RefVox] = y_ReadRPI(Ref3D, 1);
    y_Reslice(AAL, AALr, RefVox, 0, Ref3D);
  end
  y_ExtractROISignal(OutDir, {AALr}, fullfile(OutRoot, s), '', 1);
  fprintf('P6_ROI_OK %s\n', s);
end
fprintf('P6_DPARSF_DONE\n');
end


function Cfg = base_cfg(DPABIRoot, WorkDir, SubjectList)
load(fullfile(DPABIRoot, 'DPARSF', 'Jobmats', 'Template_CalculateInMNISpace_Warp_DARTEL.mat'));  % -> Cfg
Cfg.WorkingDir = WorkDir;
Cfg.DataProcessDir = WorkDir;
Cfg.SubjectID = SubjectList(:);
off = {'IsNeedConvertFunDCM2IMG', 'IsNeedConvertT1DCM2IMG', 'IsNeedReorientFunImgInteractively', ...
       'IsNeedReorientT1ImgInteractively', 'IsNeedReorientInteractivelyAfterCoreg', 'IsApplyDownloadedReorientMats', ...
       'IsBIDStoDPARSF', 'IsSmooth', 'IsCalALFF', 'IsCalReHo', 'IsCalVMHC', 'IsCalDegreeCentrality', 'IsCalFC', ...
       'IsNormalizeToSymmetricGroupT1Mean', 'IsWarpMasksIntoIndividualSpace', 'IsScrubbing', 'IsCWAS', ...
       'IsDefineROIInteractively', 'IsExtractAALTC', 'IsCalVoxelSpecificHeadMotion', 'IsDARTEL', 'IsNormalize', ...
       'IsExtractROISignals'};
for k = 1:numel(off)
  if isfield(Cfg, off{k}); Cfg.(off{k}) = 0; end
end
Cfg.TR = 0;                               % read TR from the NIfTI header
Cfg.TimePoints = 0;
Cfg.ParallelWorkersNumber = 0;
Cfg.IsAllowGUI = 0;
end
