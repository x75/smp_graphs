# check most recent run
ls -altr data/smp_expr0045_pm1d_mem000_ord0_random_infodist_id

# manually select first of most recent files and insert into cl below (FIXME)

# find all files older (-not -newer) than the file
find data/smp_expr0045_pm1d_mem000_ord0_random_infodist_id -not -newer data/smp_expr0045_pm1d_mem000_ord0_random_infodist_id/smp_expr0045_pm1d_mem000_ord0_random_infodist_id_e4d47438091226aacd935b1079df7c3c_20171226_224014_plot.pdf -exec rm -v {} \;

