name=$1
checkpoint=$2

if [ -n "$3" ]; then
    python rollouts_via_policy.py --disable_tqdm --transport $1 $2
else
    python rollouts_via_policy.py --disable_tqdm $1 $2
fi
