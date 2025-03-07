import os, subprocess

def zfp_compress_H2C(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        zfp = 'path_to_zfp_executable'
        subprocess.run(f'{zfp} -i {path_prefix}/binary_in -z {path_prefix}/zfp_out_{tol} -d -4 36 160 160 20 -a {tol} -h -s >> {path_prefix}/zfp_{tol}.txt 2>&1', shell=True)
        subprocess.run(f'{zfp} -z {path_prefix}/zfp_out_{tol} -o {path_prefix}/zfp_re_{tol} -h > /dev/null', shell=True)

def sz_compress_H2C(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        sz = 'path_to_sz_executable'
        subprocess.run(f'{sz} -z {path_prefix}/sz_out_{tol} -i {path_prefix}/binary_in -d -4 36 160 160 20 -M ABS -A {tol} > /dev/null', shell=True)
        subprocess.run(f'{sz} -x {path_prefix}/sz_re_{tol} -s {path_prefix}/sz_out_{tol} -d -4 36 160 160 20 -i {path_prefix}/binary_in -a >> {path_prefix}/sz_{tol}.txt', shell=True)

def mgard_compress_H2C(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        mgard = 'path_to_mgard_executable'
        re = r"'s/\x1b\[[0-9;]*m//g'"
        subprocess.run(f'{mgard} -z -i {path_prefix}/binary_in -c {path_prefix}/mgard_out_{tol} -t d -n 4 20 160 160 36 -m abs -e {tol} -s infinity -l 2 -d auto | sed {re} >> {path_prefix}/mgard_{tol}.txt', shell=True)
        subprocess.run(f'{mgard} -x -c {path_prefix}/mgard_out_{tol} -o {path_prefix}/mgard_re_{tol} -d auto > /dev/null', shell=True)

def zfp_compress_BF(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        if not os.path.exists(f'{path_prefix}/zfp_{tol}.txt'):
            zfp = 'path_to_zfp_executable'
            for i in range(13):
                subprocess.run(f'{zfp} -i {path_prefix}/chi_t_source/binary_in{i} -z {path_prefix}/chi_t_source/zfp_out{i}_{tol} -d -3 100 160 160 -a {tol} -h -s >> {path_prefix}/chi_t_source/zfp_{tol}.txt 2>&1', shell=True)
                subprocess.run(f'{zfp} -z {path_prefix}/chi_t_source/zfp_out{i}_{tol} -o {path_prefix}/chi_t_source/zfp_re{i}_{tol} -h > /dev/null', shell=True)

def sz_compress_BF(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        if not os.path.exists(f'{path_prefix}/sz_{tol}.txt'):
            sz = 'path_to_sz_executable'
            for i in range(13):
                subprocess.run(f'{sz} -z {path_prefix}/chi_t_source/sz_out{i}_{tol} -i {path_prefix}/chi_t_source/binary_in{i} -d -3 100 160 160 -M ABS -A {tol} > /dev/null', shell=True)
                subprocess.run(f'{sz} -x {path_prefix}/chi_t_source/sz_re{i}_{tol} -s {path_prefix}/chi_t_source/sz_out{i}_{tol}  -d -3 100 160 160 -i {path_prefix}/chi_t_source/binary_in{i} -a >> {path_prefix}/chi_t_source/sz_{tol}.txt', shell=True)

def mgard_compress_BF(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        if not os.path.exists(f'{path_prefix}/mgard_{tol}.txt'):
          mgard = 'path_to_mgard_executable'
          re = r"'s/\x1b\[[0-9;]*m//g'"
          for i in range(13):
              subprocess.run(f'{mgard} -z -i {path_prefix}/chi_t_source/binary_in{i} -c {path_prefix}/chi_t_source/mgard_out{i}_{tol} -t d -n 3 100 160 160 -m abs -e {tol} -s infinity -l 2 -d auto | sed {re} >> {path_prefix}/chi_t_source/mgard_{tol}.txt', shell=True)
              subprocess.run(f'{mgard} -x -c {path_prefix}/chi_t_source/mgard_out{i}_{tol} -o {path_prefix}/chi_t_source/mgard_re{i}_{tol} -d auto > /dev/null', shell=True)

def zfp_compress_ESAT(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        zfp = 'path_to_zfp_executable'
        subprocess.run(f'{zfp} -i {path_prefix}/binary_in -z {path_prefix}/zfp_out_{tol} -d -3 224 224 300 -a {tol} -h -s >> {path_prefix}/zfp_{tol}.txt 2>&1', shell=True)
        subprocess.run(f'{zfp} -z {path_prefix}/zfp_out_{tol} -o {path_prefix}/zfp_re_{tol} -h > /dev/null', shell=True)

def sz_compress_ESAT(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        sz = 'path_to_sz_executable'
        subprocess.run(f'{sz} -z {path_prefix}/sz_out_{tol} -i {path_prefix}/binary_in -d -3 224 224 300 -M ABS -A {tol} > /dev/null', shell=True)
        subprocess.run(f'{sz} -x {path_prefix}/sz_re_{tol} -s {path_prefix}/sz_out_{tol}  -d -3 224 224 300 -i {path_prefix}/binary_in -a >> {path_prefix}/sz_{tol}.txt', shell=True)

def mgard_compress_ESAT(tols, path_prefix='path_to_reduced_data_folder'):
    for tol in tols:
        mgard = 'path_to_mgard_executable'
        re = r"'s/\x1b\[[0-9;]*m//g'"
        subprocess.run(f'{mgard} -z -i {path_prefix}/binary_in -c {path_prefix}/mgard_out_{tol} -t d -n 3 300 224 224 -m abs -e {tol} -s infinity -l 2 -d auto | sed {re} >> {path_prefix}/mgard_{tol}.txt', shell=True)
        subprocess.run(f'{mgard} -x -c {path_prefix}/mgard_out_{tol} -o {path_prefix}/mgard_re_{tol} -d auto > /dev/null', shell=True)
