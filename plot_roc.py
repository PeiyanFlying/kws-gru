import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='ROC Plot')

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the ROC.txt',
                    default='checkpoints', type=str)

global args
args = parser.parse_args()

fp = open(args.save_dir, 'r')
lines = fp.readlines()

lines = lines[1:]
x = []
y = []
for line in lines:
	l = line.strip().split('\t')
	x.append(float(l[0]))
	y.append(float(l[1]))
plt.figure()
plt.xlim(left=0,right=1)
plt.ylim(bottom=0,top=1)
plt.plot(x,y)
plt.show()
