import os
from mgnipype import nipypes
import nipype.pipeline.engine as pe
from nipype.interfaces import spm
from nipype.interfaces.io import DataSink, DataGrabber
from nipype.interfaces.utility import IdentityInterface
import nipype.algorithms.modelgen as model
import scipy.io as sio


def l1_model_estimation(images, outputdir, conditions, timing_parameters, contrasts, rp=None, high_pass=128.,
                        concatenate_runs=False, workingdir='/data/nipypes', logging=False, autorun=False,
                        multiproc=True, keep_cache=False):
    """

    :param images (mandatory): list of Nifti objects
    :param outputdir (mandatory): output directory
    :param conditions (mandatory): conditions TODO: format
    :param timing_parameters (mandatory): timing parameters TODO: format
    :param contrasts (mandatory): contrasts TODO: format
    :param rp (optional): realignment parameters TODO: format
    :param high_pass (optional): high pass filter cutoff (float)
    :param concatenate_runs (optional): boolean
    :param workingdir (optional, default='/data/nipypes'): nipype working directory
    :param logging (optional, default=False): boolean
    :param autorun (optional, default=False): run workflow
    :param multiproc (optional, default=True): whether to use multiple processors
    :param keep_cache (optional, default=False): keep nipype cache

    :return: instance of nipype.pipeline.engine.Workflow
    """

    wf = pe.Workflow(name=os.path.basename(os.path.normpath(workingdir)))
    wf.config['execution'] = {'hash_method': 'content',  # 'timestamp' or 'content'
                              'single_thread_matlab': 'False',
                              'poll_sleep_duration': '5',
                              'stop_on_first_crash': 'False',
                              'stop_on_first_rerun': 'False'}
    if logging:
        wf.config['logging'] = {'log_directory': outputdir,
                                'log_to_file': 'True'}
    wf.base_dir = os.path.dirname(os.path.normpath(workingdir))

    modelspec = pe.Node(interface=model.SpecifySPMModel(), name="modelspec")
    modelspec.inputs.functional_runs = images
    modelspec.inputs.input_units = timing_parameters['input_units']
    modelspec.inputs.output_units = timing_parameters['output_units']
    modelspec.inputs.time_repetition = timing_parameters['time_repetition']
    modelspec.inputs.concatenate_runs = concatenate_runs
    modelspec.inputs.high_pass_filter_cutoff = high_pass
    if rp is not None:
        modelspec.inputs.realignment_parameters = rp
    modelspec.inputs.subject_info = conditions

    level1design = pe.Node(interface=spm.Level1Design(), name="level1design")
    level1design.inputs.timing_units = modelspec.inputs.output_units
    level1design.inputs.interscan_interval = modelspec.inputs.time_repetition
    level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
    level1design.inputs.microtime_resolution = timing_parameters['microtime_resolution']
    level1design.inputs.microtime_onset = timing_parameters['microtime_onset']

    level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
    level1estimate.inputs.estimation_method = {'Classical': 1}

    contrastestimate = pe.Node(interface=spm.EstimateContrast(), name="contrastestimate")
    contrastestimate.inputs.contrasts = contrasts
    contrastestimate.overwrite = True
    contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # save data
    datasink = pe.Node(DataSink(base_directory=outputdir), name="datasink")

    wf.connect([(modelspec, level1design, [('session_info', 'session_info')]),
                (level1design, level1estimate, [('spm_mat_file', 'spm_mat_file')]),
                (level1estimate, contrastestimate, [('spm_mat_file', 'spm_mat_file'),
                                                    ('beta_images', 'beta_images'),
                                                    ('residual_image', 'residual_image')]),
                (level1estimate, datasink, [('beta_images', '@beta'),
                                            ('residual_image', '@residual'),
                                            ('RPVimage', '@RPV'),
                                            ('mask_image', '@mask')]),
                (contrastestimate, datasink, [('spm_mat_file', '@SPM'),
                                              ('spmT_images', '@T'),
                                              ('con_images', '@con')
                                              ]),
                ])

    if autorun:
        if multiproc:
            import sys

            if not hasattr(sys.stdin, 'close'):
                def dummy_close():
                    pass

                sys.stdin.close = dummy_close

            wf.run('MultiProc')
        else:
            wf.run()
        print('First level statistics saved to ' + outputdir)
        print("finished!")

    if not keep_cache:
        import shutil

        shutil.rmtree(os.path.join(workingdir))

    return wf

def l1_contrast(spmmat, contrasts, workingdir='/data/nipypes', logging=False, autorun=False,
                multiproc=True, keep_cache=False):
    """

    :param spmmat (mandatory): path to SPM.mat
    :param contrasts (mandatory): contrasts TODO: format
    :param workingdir (optional, default='/data/nipypes'): nipype working directory
    :param logging (optional, default=False): boolean
    :param autorun (optional, default=False): run workflow
    :param multiproc (optional, default=True): whether to use multiple processors
    :param keep_cache (optional, default=False): keep nipype cache

    :return: instance of nipype.pipeline.engine.Workflow
    """

    wf = pe.Workflow(name=os.path.basename(os.path.normpath(workingdir)))
    wf.config['execution'] = {'hash_method': 'content',  # 'timestamp' or 'content'
                              'single_thread_matlab': 'False',
                              'poll_sleep_duration': '5',
                              'stop_on_first_crash': 'False',
                              'stop_on_first_rerun': 'False'}

    outputdir = os.path.dirname(spmmat)
    if logging:
        wf.config['logging'] = {'log_directory': outputdir,
                                'log_to_file': 'True'}
    wf.base_dir = os.path.dirname(os.path.normpath(workingdir))

    contrastestimate = pe.Node(interface=spm.EstimateContrast(), name="contrastestimate")
    contrastestimate.inputs.spm_mat_file = spmmat
    contrastestimate.inputs.contrasts = contrasts
    contrastestimate.inputs.residual_image = os.path.join(outputdir, 'ResMS.nii')
    # contrastestimate.inputs.beta_images = [os.path.join(outputdir, f) for f in sorted(os.listdir(outputdir)) if fnmatch(f, 'beta*.nii')]
    mat = sio.loadmat(spmmat)
    contrastestimate.inputs.beta_images = [os.path.join(outputdir, mat['SPM'][0][0][13][0][i][0][0]) for i in range(len(mat['SPM'][0][0][13][0]))]
    contrastestimate.overwrite = True
    contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}

    # save data
    datasink = pe.Node(DataSink(base_directory=outputdir), name="datasink")

    wf.connect([(contrastestimate, datasink, [('spm_mat_file', '@SPM'),
                                              ('spmT_images', '@T'),
                                              ('con_images', '@con')
                                              ]),
                ])

    if autorun:
        if multiproc:
            import sys

            if not hasattr(sys.stdin, 'close'):
                def dummy_close():
                    pass

                sys.stdin.close = dummy_close

            wf.run('MultiProc')
        else:
            wf.run()
        print('First level statistics saved to ' + outputdir)
        print("finished!")

    if not keep_cache:
        import shutil

        shutil.rmtree(os.path.join(workingdir))

    return wf



def normalize_smooth(path_images, subjects=None, deformation=None, structural=None, smoothing=None, save_normalized=True,
                     workingdir='/tmp/nipype/', logging=False, autorun=True, multiproc=True, keep_cache=False):
    """

    Args:
        path_images (mandatory): a string that specifies the path to a Nifti image, a list of strings that specify the
                                 path to a Nifti image, or a dict with fields 'dir' and 'template' if nipype's iterating
                                 procedure is used. The value of 'dir' must then correspond to the base directory
                                 containing the subject data and 'template' specifies the path after 'dir' that leads to
                                 the individual subject data.
        subjects (optional): if a list with subject id's is passed, nipype's iterating procedure is used. In this case,
                             path_images and either deformation or structural must be a dict.
        deformation (optional): path to an SPM deformation file or a dict with fields 'dir' and 'template'. If a dict is
                                passed, nipype's iterating procedure is used. The value of 'dir' specifies the base
                                directory containing the subject's anatomical data and 'template' specifies the path
                                after dir that leads to the subject's individual deformation file.
        structural (optional): path to a structural file or a dict with fields 'dir' and 'template'. If a dict is
                               passed, nipype's iterating procedure is used. The value of 'dir' specifies the base
                               directory containing the subject's anatomical data and 'template' specifies the path
                               after dir that leads to the subject's individual structural file.
        smoothing (optional): isotropic smoothing kernel in mm
        save_normalized (optional): whether to save the normalized data (set to False, if a smoothing kernel is passed
                                    and you wish to not keep the normalized files).
        workingdir (optional): nipype's working directory
        logging:
        autorun:
        multiproc:
        keep_cache:

    Returns:

        A nipype workflow structure.

    """
    if not isinstance(path_images, list) and subjects is None:
        path_images = [path_images]

    if subjects is not None:
        if not isinstance(path_images, dict):
            raise ValueError('If variable subjects is defined, path images must be a dict.')

        infosource, datasource, anatsource = _data_grabber(path_images, deformation, subjects)

    if deformation is not None and structural is not None:
        raise ValueError('Either provide a deformation file or a structural image, but not both!')
    if deformation is None and structural is None:
        raise ValueError('Either provide a deformation file or a structural image!')
    if smoothing is None and not save_normalized:
        raise ValueError("Not generating smoothed files *and* not saving normalized files doesn't make sense!")

    wf = pe.Workflow(name=os.path.basename(os.path.normpath(workingdir)))
    wf.config['execution'] = {'hash_method': 'content',  # 'timestamp' or 'content'
                              'single_thread_matlab': 'False',
                              'poll_sleep_duration': '5',
                              'stop_on_first_crash': 'False',
                              'stop_on_first_rerun': 'False'}

    if subjects is None:
        outputdir = os.path.dirname(path_images[0])
    else:
        outputdir = path_images['dir']

    if logging:
        wf.config['logging'] = {'log_directory': outputdir,
                                'log_to_file': 'True'}
    wf.base_dir = os.path.dirname(os.path.normpath(workingdir))

    # normalization
    nm = pe.Node(interface=spm.Normalize12(), name="normalize")
    if subjects is None:
        nm.inputs.apply_to_files = path_images
    if deformation is not None:
        nm.inputs.jobtype = 'write'
        if subjects is None:
            nm.inputs.deformation_file = deformation
    if structural is not None:
        nm.inputs.jobtype = 'estwrite'
        if subjects is None:
            nm.inputs.image_to_align = structural
        nm.inputs.tpm = nipypes.TPM

    # smoothing
    if smoothing is not None:
        smooth = pe.Node(interface=spm.Smooth(), name="smooth")
        smooth.inputs.out_prefix = 's%g' % smoothing
        smooth.inputs.fwhm = [smoothing, smoothing, smoothing]

    # save data
    datasink = pe.Node(DataSink(base_directory=outputdir), name="datasink")
    datasink.inputs.parameterization = False

    links = []
    if subjects is not None:
        links.append((infosource, datasource, [('subject_id', 'subject_id')]))
        links.append((infosource, anatsource, [('subject_id', 'subject_id')]))
        links.append((datasource, nm, [('data', 'apply_to_files')]))
        if deformation is not None:
            links.append((anatsource, nm, [('data', 'deformation_file')]))
        elif structural is not None:
            links.append((anatsource, nm, [('data', 'image_to_align')]))
    if save_normalized:
        links.append((nm, datasink, [('normalized_files', '@norm')]))
    if structural is not None:
        links.append((nm, datasink, [('deformation_field', '@deform')]))
    if smoothing is not None:
        links.append((nm, smooth, [('normalized_files', 'in_files')]))
        links.append((smooth, datasink, [('smoothed_files', '@smooth')]))
    wf.connect(links)

    wf.write_graph(graph2use='colored', dotfilename='/home/matteo/graph.dot')

    if autorun:
        if multiproc:
            import sys

            if not hasattr(sys.stdin, 'close'):
                def dummy_close():
                    pass

                sys.stdin.close = dummy_close

            wf.run('MultiProc')
        else:
            wf.run()
        print("finished!")

    if not keep_cache:
        import shutil
        shutil.rmtree(os.path.join(workingdir))

    return wf


def _data_grabber(data_info, anat_info, subjects):

    # Map field names to individual subject runs.
    info = dict(data=[['subject_id']])

    infosource = pe.Node(interface=IdentityInterface(fields=['subject_id']), name="infosource")
    infosource.iterables = ('subject_id', subjects)

    datasource = pe.Node(interface=DataGrabber(infields=['subject_id'], outfields=['data']), name='datasource')
    datasource.inputs.base_directory = data_info['dir']
    datasource.inputs.template = data_info['template']
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True

    anatsource = pe.Node(interface=DataGrabber(infields=['subject_id'], outfields=['data']), name='anatsource')
    anatsource.inputs.base_directory = anat_info['dir']
    anatsource.inputs.template = anat_info['template']
    anatsource.inputs.template_args = info
    anatsource.inputs.sort_filelist = True

    return infosource, datasource, anatsource



def l2_one_sample_ttest(images, outputdir, explicit_mask=False, effect_name='effect', workingdir='/data/nipypes',
                        covariates=None, logging=False, autorun=False, multiproc=True, keep_cache=False):
    """

    :param images (mandatory): list of Nifti objects
    :param labels (mandatory): numpy array of 1's and 2's
    :param outputdir (mandatory'): output directory
    :param group_names (optional, default=('group 1', 'group 2')): tuple with group names
    :param workingdir (optional, default='/data/nipypes'): nipype working directory
    :param covariates (optional, default=None): list of covariate dictionaries
    :param logging (optional, default=False): boolean
    :param autorun (optional, default=False): run workflow
    :param keep_cache (optional, default=False): keep nipype cache
    :return: instance of nipype.pipeline.engine.Workflow
    """

    if not os.path.exists(outputdir):
        print("Creating output directory % s" % outputdir)
        os.mkdir(outputdir)

    # workflow
    wf = pe.Workflow(name=os.path.basename(os.path.normpath(workingdir)))
    wf.config['execution'] = {'hash_method': 'content',  # 'timestamp' or 'content'
                              'single_thread_matlab': 'False',
                              'poll_sleep_duration': '5',
                              'stop_on_first_crash': 'False',
                              'stop_on_first_rerun': 'False'}
    if logging:
        wf.config['logging'] = {'log_directory': outputdir,
                                'log_to_file': 'True'}
    wf.base_dir = os.path.dirname(os.path.normpath(workingdir))

    # create mask if it does not exist
    if explicit_mask is True:
        if not os.path.exists(os.path.join(outputdir, 'mask.nii')):
            print("Creating mask file% s" % os.path.join(outputdir, 'mask.nii'))
            from nilearn.masking import compute_multi_epi_mask, compute_background_mask
            import nibabel

            mask = compute_multi_epi_mask(images)
            # mask = compute_background_mask(images)
            mask.set_data_dtype(float)
            mask_path = os.path.join(outputdir, 'mask.nii')
            nibabel.save(mask, mask_path)
    elif isinstance(explicit_mask, str) and os.path.exists(explicit_mask):
        mask_path = explicit_mask
    elif explicit_mask is not None and explicit_mask is not False:
        raise ValueError('Invalid value for explicit_mask!')


    # model design
    ttest_design = pe.Node(interface=spm.OneSampleTTestDesign(), name="ttest_design")
    ttest_design.inputs.in_files = [img for i, img in enumerate(images)]
    ttest_design.inputs.threshold_mask_none = True
    ttest_design.inputs.use_implicit_threshold = True
    if explicit_mask:
        ttest_design.inputs.explicit_mask_file = mask_path
    if covariates is not None:
        ttest_design.inputs.covariates = covariates

    # model estimation
    ttest_estimate = pe.Node(interface=spm.EstimateModel(), name="ttest_estimate")
    ttest_estimate.inputs.estimation_method = {'Classical': 1}

    # contrasts
    ttest_contrast = pe.Node(interface=spm.EstimateContrast(), name="ttest_contrast")
    con_1 = ('+ (%s)' % effect_name, 'T', ['mean'], [1])
    con_2 = ('- (%s)' % effect_name, 'T', ['mean'], [-1])
    ttest_contrast.inputs.contrasts = [con_1, con_2]
    if covariates is not None:
        for cov in covariates:
            ttest_contrast.inputs.contrasts += [('+' + cov['name'], 'T', [cov['name']], [1])]
            ttest_contrast.inputs.contrasts += [('-' + cov['name'], 'T', [cov['name']], [-1])]
    ttest_contrast.inputs.group_contrast = True

    # save data
    datasink = pe.Node(DataSink(base_directory=outputdir), name="datasink")

    # connect nodes
    wf.connect([(ttest_design, ttest_estimate, [('spm_mat_file', 'spm_mat_file')]),
                (ttest_estimate, ttest_contrast, [('spm_mat_file', 'spm_mat_file'),
                                                  ('beta_images', 'beta_images'),
                                                  ('residual_image', 'residual_image')]),
                (ttest_estimate, datasink, [('residual_image', '@res'),
                                            ('mask_image', '@mask'),
                                            ('beta_images', '@beta'),
                                            ('RPVimage', '@RPV')]),
                (ttest_contrast, datasink, [('spm_mat_file', '@SPM'),
                                            ('spmT_images', '@T'),
                                            ('con_images', '@con')]),
                ])

    if autorun:
        if multiproc:
            import sys

            if not hasattr(sys.stdin, 'close'):
                def dummy_close():
                    pass

                sys.stdin.close = dummy_close

            wf.run('MultiProc')
        else:
            wf.run()
        print('Group level statistics saved to ' + outputdir)
        print("finished!")

    if not keep_cache:
        import shutil

        shutil.rmtree(os.path.join(workingdir))

    return wf

def l2_two_sample_ttest(images, labels, outputdir, group_names=('group 1', 'group 2'), workingdir='/data/nipypes',
                        covariates=None, logging=False, autorun=False, multiproc=True, keep_cache=False):
    """

    :param images (mandatory): list of Nifti objects
    :param labels (mandatory): numpy array of 1's and 2's
    :param outputdir (mandatory'): output directory
    :param group_names (optional, default=('group 1', 'group 2')): tuple with group names
    :param workingdir (optional, default='/data/nipypes'): nipype working directory
    :param covariates (optional, default=None): list of covariate dictionaries
    :param logging (optional, default=False): boolean
    :param autorun (optional, default=False): run workflow
    :param keep_cache (optional, default=False): keep nipype cache
    :return: instance of nipype.pipeline.engine.Workflow
    """

    if not os.path.exists(outputdir):
        print("Creating output directory % s" % outputdir)
        os.mkdir(outputdir)

    # workflow
    wf = pe.Workflow(name=os.path.basename(os.path.normpath(workingdir)))
    wf.config['execution'] = {'hash_method': 'content',  # 'timestamp' or 'content'
                              'single_thread_matlab': 'False',
                              'poll_sleep_duration': '5',
                              'stop_on_first_crash': 'False',
                              'stop_on_first_rerun': 'False'}
    if logging:
        wf.config['logging'] = {'log_directory': outputdir,
                                'log_to_file': 'True'}
    wf.base_dir = os.path.dirname(os.path.normpath(workingdir))

    # create mask if it does not exist
    if not os.path.exists(os.path.join(outputdir, 'mask.nii')):
        print("Creating mask file% s" % os.path.join(outputdir, 'mask.nii'))
        from nilearn.masking import compute_multi_epi_mask
        import nibabel

        mask = compute_multi_epi_mask(images)
        mask.set_data_dtype(float)
        nibabel.save(mask, os.path.join(outputdir, 'mask.nii'))

    # model design
    ttest_design = pe.Node(interface=spm.TwoSampleTTestDesign(), name="ttest_design")
    ttest_design.inputs.group1_files = [img for i, img in enumerate(images) if labels[i] == 1]
    ttest_design.inputs.group2_files = [img for i, img in enumerate(images) if labels[i] == 2]
    ttest_design.inputs.explicit_mask_file = os.path.join(outputdir, 'mask.nii')
    if covariates is not None:
        ttest_design.inputs.covariates = covariates
    ttest_design.inputs.threshold_mask_none = True
    ttest_design.inputs.group_contrast = True

    # model estimation
    ttest_estimate = pe.Node(interface=spm.EstimateModel(), name="ttest_estimate")
    ttest_estimate.inputs.estimation_method = {'Classical': 1}

    # contrasts
    ttest_contrast = pe.Node(interface=spm.EstimateContrast(), name="ttest_contrast")
    con_1 = (group_names[0], 'T', ['Group_{1}'], [1])
    con_2 = (group_names[1], 'T', ['Group_{2}'], [1])
    con_3 = ('%s > %s' % tuple(group_names), 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
    con_4 = ('%s < %s' % tuple(group_names), 'T', ['Group_{1}', 'Group_{2}'], [-1, 1])
    ttest_contrast.inputs.contrasts = [con_1, con_2, con_3, con_4]
    if covariates is not None:
        for cov in covariates:
            ttest_contrast.inputs.contrasts += [('+' + cov['name'], 'T', [cov['name']], [1])]
            ttest_contrast.inputs.contrasts += [('-' + cov['name'], 'T', [cov['name']], [-1])]

    # save data
    datasink = pe.Node(DataSink(base_directory=outputdir), name="datasink")

    # connect nodes
    wf.connect([(ttest_design, ttest_estimate, [('spm_mat_file', 'spm_mat_file')]),
                (ttest_estimate, ttest_contrast, [('spm_mat_file', 'spm_mat_file'),
                                                  ('beta_images', 'beta_images'),
                                                  ('residual_image', 'residual_image')]),
                (ttest_estimate, datasink, [('residual_image', '@res'),
                                            ('beta_images', '@beta'),
                                            ('RPVimage', '@RPV')]),
                (ttest_contrast, datasink, [('spm_mat_file', '@SPM'),
                                            ('spmT_images', '@T'),
                                            ('con_images', '@con')]),
                ])

    if autorun:
        if multiproc:
            import sys

            if not hasattr(sys.stdin, 'close'):
                def dummy_close():
                    pass

                sys.stdin.close = dummy_close

            wf.run('MultiProc')
        else:
            wf.run()
        print('Group level statistics saved to ' + outputdir)
        print("finished!")

    if not keep_cache:
        import shutil

        shutil.rmtree(os.path.join(workingdir))

    return wf
