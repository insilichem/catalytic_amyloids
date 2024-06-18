import fibTool_v3 as fib
a=fib.gen_fibril(rep_intra=20,offset_AP=-1.0,refpdb="REFERENCE.pdb",typefibril='Pffuu', intraprotodist=5.0, interprotodist=9.0,n_layers=4,interlayer_offset=4.)
names,coord=a.readcoordinates()
a.makefibril(names,coord)
a=fib.gen_fibril(rep_intra=20,offset_AP=-1.0,refpdb="REFERENCE.pdb",typefibril='Pfbuu', intraprotodist=5.0, interprotodist=7.0,n_layers=4,interlayer_offset=4.)
names,coord=a.readcoordinates()
a.makefibril(names,coord)
a=fib.gen_fibril(rep_intra=20,offset_AP=-1.0,refpdb="REFERENCE.pdb",typefibril='Pfbud', intraprotodist=5.0, interprotodist=9.0,n_layers=4,interlayer_offset=4.)
names,coord=a.readcoordinates()
a.makefibril(names,coord)
a=fib.gen_fibril(rep_intra=20,offset_AP=-1.0,refpdb="REFERENCE.pdb",typefibril='Pffud', intraprotodist=5.0, interprotodist=9.0,n_layers=4,interlayer_offset=4.)
names,coord=a.readcoordinates()
a.makefibril(names,coord)
