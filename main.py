import sys
from util.input_args import get_args
from models.simclr import SimCLR
from util.dataloader import Dataloader
import numpy as np

args = get_args(sys.argv[1:])
dataloader = Dataloader(args)
model = SimCLR(args)
print(model.model.summary())
# a = model.model.predict(dataloader[0][0][:1])
model.train(dataloader)

