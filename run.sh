
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 18 --first-channel 16 --repeat 0 --red-times 1 --channel-inc 2 --id "channel/16"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 18 --first-channel 12 --repeat 0 --red-times 1 --channel-inc 2 --id "channel/12"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 18 --first-channel 4 --repeat 0 --red-times 1 --channel-inc 2 --id "channel/4"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 18 --first-channel 2 --repeat 0 --red-times 1 --channel-inc 2 --id "channel/2"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 18 --first-channel 1 --repeat 0 --red-times 1 --channel-inc 2 --id "channel/1"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 36 --first-channel 8 --repeat 0 --red-times 1 --channel-inc 2 --id "latent"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 54 --first-channel 8 --repeat 0 --red-times 1 --channel-inc 2 --id "latent"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 72 --first-channel 8 --repeat 0 --red-times 1 --channel-inc 2 --id "latent"
  if [$?]; then
    break
  fi
done
while :
do
  python train_InceptionRedVAE.py --epochs 100 --batch-size 1024 --channels 1 64 --cnn-outsize 34560 --latent 18 --first-channel 8 --repeat 0 --red-times 1 --channel-inc 2 --id "standard"
  if [$?]; then
    break
  fi
done