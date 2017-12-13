# load a cProfile output file
p_noafm2 = pstats.Stats('experiment_expr0060_noafm2.prof')

# clean up, sort, and print top 30
p_noafm2.strip_dirs().sort_stats('tottime').print_stats(30)

# print something like
"""
Tue Nov 28 12:06:18 2017    experiment_expr0060_noafm2.prof

         23451347 function calls (23235502 primitive calls) in 12.739 seconds

   Ordered by: internal time
   List reduced from 14189 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      149    0.892    0.006    2.696    0.018 afm.py:218(_parse_kern_pairs)
  1469619    0.623    0.000    0.623    0.000 {_codecs.utf_8_decode}
  1460452    0.567    0.000    0.733    0.000 graph.py:626(nodes)
      180    0.535    0.003    1.212    0.007 afm.py:178(_parse_char_metrics)
  1467030    0.462    0.000    1.390    0.000 {method 'decode' of 'str' objects}
  1469287    0.307    0.000    0.929    0.000 utf_8.py:15(decode)
   100004    0.281    0.000    0.615    0.000 graph.py:404(<genexpr>)
  1465819    0.262    0.000    1.649    0.000 afm.py:64(_to_str)
18092/18046    0.253    0.000    0.364    0.000 numeric.py:1343(roll)
    64000    0.248    0.000    0.546    0.000 graph.py:391(<genexpr>)
    18000    0.209    0.000    2.708    0.000 block.py:622(process_input)
59002/41002    0.170    0.000    1.319    0.000 graph.py:396(nxgraph_node_by_id_recursive)
  1460452    0.167    0.000    0.167    0.000 reportviews.py:167(__init__)
   517988    0.161    0.000    0.258    0.000 afm.py:198(<genexpr>)
   744029    0.159    0.000    0.159    0.000 {method 'split' of 'str' objects}
   233114    0.148    0.000    0.309    0.000 graph.py:69(<lambda>)
  1440432    0.146    0.000    0.146    0.000 reportviews.py:177(__getitem__)
    94748    0.139    0.000    0.141    0.000 {numpy.core.multiarray.array}
      122    0.125    0.001    0.125    0.001 {cPickle.loads}
   627626    0.122    0.000    0.122    0.000 {method 'split' of 'unicode' objects}
   103455    0.121    0.000    0.146    0.000 afm.py:73(_to_list_of_floats)
        2    0.116    0.058    4.068    2.034 font_manager.py:544(createFontList)
    27142    0.109    0.000    0.215    0.000 colors.py:142(_to_rgba_no_colorcycle)
    41000    0.093    0.000    2.093    0.000 block.py:479(get_blocksize_input)
    34218    0.092    0.000    0.116    0.000 {method 'sub' of '_sre.SRE_Pattern' objects}
        2    0.091    0.046    0.094    0.047 measures_infth.py:183(infth_mi_multivariate)
   870276    0.088    0.000    0.088    0.000 {method 'startswith' of 'str' objects}
    22165    0.082    0.000    0.082    0.000 {method 'reduce' of 'numpy.ufunc' objects}
        2    0.074    0.037    0.074    0.037 backend_qt5.py:101(_create_qApp)
      480    0.073    0.000    0.082    0.000 {method '_create_array' of 'tables.hdf5extension.Array' objects}
"""

"""
 - afm is matplotlib adobe font metrics
 - decode calls are from stdout printing / logging?
 - graph: check nodes and genexpr
"""


