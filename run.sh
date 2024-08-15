### toy example
# python sivistein_2d.py --config "multimodal.yml" --log_stick
# python sivistein_2d.py --config "x_shaped.yml" --log_stick
# python sivistein_2d.py --config "banana.yml" --log_stick
# python sivistein_t_student.py --config "student_uc.yml" --log_stick

### lr
# python sivistein_lr.py --config LRwaveform.yml --baseline_sample SGLD_trace/parallel_SGLD_LRwaveform.pt --log_stick

### cd
# python sivistein_langevin_post.py --config kernel_sivi_langevin_post.yml --log_stick 
# python sgld_langevin_post.py

### bnn
# python sivistein_bnn.py --config kernel_sivi_boston.yml --log_stick
# python sivistein_bnn.py --config kernel_sivi_concrete.yml --log_stick
# python sivistein_bnn.py --config kernel_sivi_power.yml --log_stick
# python sivistein_bnn.py --config kernel_sivi_protein.yml --log_stick
# python sivistein_bnn.py --config kernel_sivi_winered.yml --log_stick
# python sivistein_bnn.py --config kernel_sivi_yacht.yml --log_stick
