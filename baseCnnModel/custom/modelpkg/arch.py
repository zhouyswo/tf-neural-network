import baseCnnModel.custom.modelpkg.architectures.alexnet as alexnet
import baseCnnModel.custom.modelpkg.architectures.resnet as resnet
import baseCnnModel.custom.modelpkg.architectures.vgg as vgg
import baseCnnModel.custom.modelpkg.architectures.googlenet as googlenet
import baseCnnModel.custom.modelpkg.architectures.nin as nin
import baseCnnModel.custom.modelpkg.architectures.densenet

def get_model(inputs, wd, is_training, args, transferMode= False):
    if args.architecture=='alexnet':
        return architectures.alexnet.inference(inputs, args.num_classes, wd, 0.5 if is_training else 1.0, is_training, transferMode)
    elif args.architecture=='resnet':
        return architectures.resnet.inference(inputs, args.depth, args.num_classes, wd, is_training, transferMode)
    elif args.architecture=='densenet':
        return architectures.densenet.inference(inputs, args.depth, args.num_classes, wd, is_training, transferMode)
    elif args.architecture=='vgg':
        return architectures.vgg.inference(inputs, args.num_classes, wd, 0.5 if is_training else 1.0, is_training, transferMode)
    elif args.architecture=='googlenet':
        return architectures.googlenet.inference(inputs, args.num_classes, wd, 0.4 if is_training else 1.0, is_training, transferMode)
    elif args.architecture=='nin':
        return architectures.nin.inference(inputs, args.num_classes, wd, is_training, transferMode)
