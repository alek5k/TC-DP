checkpoint=$2
name=$1

if [ -n "$3" ]; then
    python gather_rollouts.py -c $2 -n 400 -o "rollouts/$1" -e $3
else
    python gather_rollouts.py -c $2 -n 400 -o "rollouts/$1"
fi