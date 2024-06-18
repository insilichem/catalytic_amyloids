# By F. Peccati, last update 05/05/2021
import numpy as np
import math as mt

# Definitions
# Rotation angle and rotation matrices
rot_angle = np.pi

# Rotation matrix around the x axis
rot_matrixx = np.zeros(shape=(3, 3))
rot_matrixx[0, 0] = 1.
rot_matrixx[0, 1] = 0.
rot_matrixx[0, 2] = 0.
rot_matrixx[1, 0] = 0.
rot_matrixx[1, 1] = mt.cos(rot_angle)
rot_matrixx[1, 2] = -mt.sin(rot_angle)
rot_matrixx[2, 0] = 0.
rot_matrixx[2, 1] = mt.sin(rot_angle)
rot_matrixx[2, 2] = mt.cos(rot_angle)

# Rotation mtrix around the y axis
rot_matrixy = np.zeros(shape=(3, 3))
rot_matrixy[0, 0] = mt.cos(rot_angle)
rot_matrixy[0, 1] = 0.
rot_matrixy[0, 2] = mt.sin(rot_angle)
rot_matrixy[1, 0] = 0.
rot_matrixy[1, 1] = 1.
rot_matrixy[1, 2] = 0.
rot_matrixy[2, 0] = -mt.sin(rot_angle)
rot_matrixy[2, 1] = 0.
rot_matrixy[2, 2] = mt.cos(rot_angle)

# Rotation matrix around the z axis
rot_matrixz = np.zeros(shape=(3, 3))
rot_matrixz[0, 0] = mt.cos(rot_angle)
rot_matrixz[0, 1] = -mt.sin(rot_angle)
rot_matrixz[0, 2] = 0.
rot_matrixz[1, 0] = mt.sin(rot_angle)
rot_matrixz[1, 1] = mt.cos(rot_angle)
rot_matrixz[1, 2] = 0.
rot_matrixz[2, 0] = 0.
rot_matrixz[2, 1] = 0.
rot_matrixz[2, 2] = 1.

# Class definition with 2 static methods: one for reading the initial PDB information (readcoordinates) and one for generating the fibrils (makefibril)


class gen_fibril:
    """This class builds fibril models from single strands using geometric considerations.

       This class has two tatic methods, one for reading the initial PDB information 
       (readcoordinates) and one for generating the fibrils (makefibril)
       """

    def __init__(self,
                 refpdb='refpdb',
                 offset_AP=-1.0,
                 offset_z=2.3,
                 offset_y=2.3,
                 interlayer_offset=0.,
                 typefibril='Pfbuu',
                 intraprotodist=4.6,
                 n_layers=2.,
                 rep_intra=3,
                 interprotodist=5.0):
        self.refpdb = refpdb
        self.typefibril = typefibril
        self.offset_AP = offset_AP
        self.interlayer_offset = interlayer_offset
        self.n_layers = n_layers
        self.offset_z = offset_z
        self.offset_y = offset_y
        self.intraprotodist = intraprotodist
        self.interprotodist = interprotodist
        self.rep_intra = rep_intra

    def readcoordinates(self):
        """Cartesian coordinates of the atoms in REFERENCE.pdb are read
          """

        with open("%s" % self.refpdb, "r") as f:
            lines = f.read().splitlines()
            atoms_info = []
            for i in range(0, len(lines)):
                if lines[i].strip() in ['TER', 'END']:
                    pass
                else:
                    atoms_info.append(lines[i])
            HETorAT = []
            index = []
            atname = []
            chain = []
            resid = []
            resname = []
            x_coord = []
            y_coord = []
            z_coord = []
            x_coord_float = []
            y_coord_float = []
            z_coord_float = []
            preinfo = []
            for i in range(0, len(atoms_info)):
                HETorAT.append(atoms_info[i].strip()[0:6])
                index.append(atoms_info[i].strip()[6:11])
                atname.append(atoms_info[i].strip()[11:16])
                resname.append(atoms_info[i].strip()[16:20])
                chain.append(atoms_info[i].strip()[20:22])
                resid.append(atoms_info[i].strip()[22:26])
                x_coord.append(atoms_info[i].strip()[26:38])
                y_coord.append(atoms_info[i].strip()[38:46])
                z_coord.append(atoms_info[i].strip()[46:54])
                preinfo.append(atoms_info[i].strip()[0:26])
            for i in range(0, len(atoms_info)):
                x_coord_float.append(float(x_coord[i]))
                y_coord_float.append(float(y_coord[i]))
                z_coord_float.append(float(z_coord[i]))
            coordinates = np.array(
                [x_coord_float, y_coord_float, z_coord_float])
            return preinfo, coordinates

    def makefibril(self, preinfo, coordinates):
        """Based on the fibril type, rotations and translations are used to generete the fibril models

           The following types are available:
               'Pffuu' 
               'APfeqbuu' 
               'APfeqbud'
               'APffueqd'
               'Pfbuu'
               'Pfbud'
               'APfbueqd'
               'Pffud'
          """

        if self.typefibril is 'Pffuu':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixy, coordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            with open("output_Pff1uu_interlayer_offset1.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist-self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_Pff1uu_interlayer_offset2.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist+self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_Pff2uu_interlayer_offset1.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist-self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
            with open("output_Pff2uu_interlayer_offset2.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist+self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        # inizia il nuovo blocco
        elif self.typefibril is 'APfeqbuu':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixy, coordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            with open("output_AP1feqbuu.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_AP+self.offset_z
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = coordinates[2, j] + \
                                    self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
            with open("output_AP2feqbuu.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_AP+self.offset_z
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = coordinates[2, j] + \
                                    self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        elif self.typefibril is 'APfeqbud':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            int2coordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            int3coordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixx, coordinates[:, j])
                int2coordinates[:, j] = np.matmul(
                    rot_matrixy, coordinates[:, j])
                int3coordinates[:, j] = np.matmul(
                    rot_matrixz, coordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            center3x = np.average(int2coordinates[0, :])
            center3y = np.average(int2coordinates[1, :])
            center3z = np.average(int2coordinates[2, :])
            center4x = np.average(int3coordinates[0, :])
            center4y = np.average(int3coordinates[1, :])
            center4z = np.average(int3coordinates[2, :])
            with open("output_AP1feqbud.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_AP+self.offset_z
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int2coordinates[0, j]-(center3x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    int2coordinates[1, j]-(center3y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int2coordinates[2, j]-(center3z-center1z))
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_AP2feqbud.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_AP+self.offset_z
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int2coordinates[0, j]-(center3x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    int2coordinates[1, j]-(center3y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int2coordinates[2, j]-(center3z-center1z))
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        elif self.typefibril is 'APffueqd':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            int2coordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            int3coordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixy, coordinates[:, j])
                int2coordinates[:, j] = np.matmul(
                    rot_matrixz, coordinates[:, j])
            for j in range(0, np.shape(coordinates)[1]):
                int3coordinates[:, j] = np.matmul(
                    rot_matrixz, intcoordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            center3x = np.average(int2coordinates[0, :])
            center3y = np.average(int2coordinates[1, :])
            center3z = np.average(int2coordinates[2, :])
            center4x = np.average(int3coordinates[0, :])
            center4y = np.average(int3coordinates[1, :])
            center4z = np.average(int3coordinates[2, :])
            with open("output_APff1ueqd_interlayer_offset1.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int2coordinates[0, j]-(center3x-center1x))+k*self.interprotodist - self.interlayer_offset
                                newcoords[1, j] = (
                                    int2coordinates[1, j]-(center3y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int2coordinates[2, j]-(center3z-center1z))+self.offset_AP
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist - self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_APff1ueqd_interlayer_offset2.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int2coordinates[0, j]-(center3x-center1x))+k*self.interprotodist + self.interlayer_offset
                                newcoords[1, j] = (
                                    int2coordinates[1, j]-(center3y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int2coordinates[2, j]-(center3z-center1z))+self.offset_AP
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist + self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_APff2ueqd_interlayer_offset1.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int2coordinates[0, j]-(center3x-center1x))-k*self.interprotodist-self.interlayer_offset
                                newcoords[1, j] = (
                                    int2coordinates[1, j]-(center3y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int2coordinates[2, j]-(center3z-center1z))+self.offset_AP
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist-self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_APff2ueqd_interlayer_offset2.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int2coordinates[0, j]-(center3x-center1x))-k*self.interprotodist+self.interlayer_offset
                                newcoords[1, j] = (
                                    int2coordinates[1, j]-(center3y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int2coordinates[2, j]-(center3z-center1z))+self.offset_AP
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist+self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        elif self.typefibril is 'Pfbuu':
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            with open("output_P1fbuu.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = coordinates[2, j] + \
                                    self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_P2fbuu.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = coordinates[2, j] + \
                                    self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        elif self.typefibril is 'Pfbud':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixx, coordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            with open("output_Pfb1ud.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_Pfb2ud.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        elif self.typefibril is 'APfbueqd':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            int2coordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            int3coordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixy, coordinates[:, j])
                int2coordinates[:, j] = np.matmul(
                    rot_matrixz, coordinates[:, j])
            for j in range(0, np.shape(coordinates)[1]):
                int3coordinates[:, j] = np.matmul(
                    rot_matrixz, intcoordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            center3x = np.average(int2coordinates[0, :])
            center3y = np.average(int2coordinates[1, :])
            center3z = np.average(int2coordinates[2, :])
            center4x = np.average(int3coordinates[0, :])
            center4y = np.average(int3coordinates[1, :])
            center4z = np.average(int3coordinates[2, :])
            with open("output_APfb1ueqd.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = coordinates[2, j] + \
                                    self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))+k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_APfb2ueqd.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            elif k % 2 != 0 and i % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = coordinates[2, j] + \
                                    self.offset_z
                            elif k % 2 == 0 and i % 2 != 0:
                                newcoords[0, j] = (
                                    int3coordinates[0, j]-(center4x-center1x))-k*self.interprotodist
                                newcoords[1, j] = (
                                    int3coordinates[1, j]-(center4y-center1y))+i*self.intraprotodist
                                newcoords[2, j] = (
                                    int3coordinates[2, j]-(center4z-center1z))+self.offset_AP
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
        elif self.typefibril is 'Pffud':
            center1x = np.average(coordinates[0, :])
            center1y = np.average(coordinates[1, :])
            center1z = np.average(coordinates[2, :])
            newcoords = np.zeros(shape=(3, np.shape(coordinates)[1]))
            intcoordinates = np.zeros(shape=(3, np.shape(coordinates)[1]))
            for j in range(0, np.shape(coordinates)[1]):
                intcoordinates[:, j] = np.matmul(
                    rot_matrixz, coordinates[:, j])
            center2x = np.average(intcoordinates[0, :])
            center2y = np.average(intcoordinates[1, :])
            center2z = np.average(intcoordinates[2, :])
            with open("output_Pff1ud_interlayer_offset1.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist-self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_Pff1ud_interlayer_offset2.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] + \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))+k*self.interprotodist+self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_Pff2ud_interlayer_offset1.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist - self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
            with open("output_Pff2ud_interlayer_offset2.pdb", "w") as h:
                for i in range(0, self.rep_intra):
                    for k in range(0, self.n_layers):
                        for j in range(0, np.shape(coordinates)[1]):
                            if k % 2 == 0:
                                newcoords[0, j] = coordinates[0, j] - \
                                    k*self.interprotodist
                                newcoords[1, j] = coordinates[1, j] + \
                                    i*self.intraprotodist
                                newcoords[2, j] = coordinates[2, j]
                            elif k % 2 != 0:
                                newcoords[0, j] = (
                                    intcoordinates[0, j]-(center2x-center1x))-k*self.interprotodist + self.interlayer_offset
                                newcoords[1, j] = (
                                    intcoordinates[1, j]-(center2y-center1y))+i*self.intraprotodist+self.offset_y
                                newcoords[2, j] = (
                                    intcoordinates[2, j]-(center2z-center1z))+self.offset_z
                            h.write(
                                preinfo[j]+"{:12.3f}".format(newcoords[0, j]))
                            h.write("{:8.3f}".format(newcoords[1, j]))
                            h.write("{:8.3f}".format(newcoords[2, j]))
                            h.write("\n")
                        h.write("TER")
                        h.write("\n")
                h.write("END")
