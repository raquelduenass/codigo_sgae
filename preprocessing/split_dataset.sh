##!/bin/bash

# Portion of training: 1/RANGE will be in testing
VAL_RANGE=5
TEST_RANGE=5

val_dir="/mnt/HD16TB/databases/tmp/validation"
test_dir="/mnt/HD16TB/databases/tmp/testing"
train_dir="/mnt/HD16TB/databases/tmp/training"

number=$RANDOM
echo "number %= $RANGE"


find "$train_dir" -maxdepth 1 -mindepth 1 -type d | while read user;
do
   find "$user" -maxdepth 1 -mindepth 1 -type d | while read class;
   do
      mkdir -p "$test_dir/$(basename "$user")/$(basename "$class")"
      find "$class" -maxdepth 1 -mindepth 1 -type f | while read image;
      do
         number=$RANDOM
	 let "number %= $TEST_RANGE"
	 if [ "$number" -lt "1" ]
	 then
	    echo "Moved $image"
	    mv "$image" "$test_dir/$(basename "$user")/$(basename "$class")"
	 fi
      done
   done
done


find "$train_dir" -maxdepth 1 -mindepth 1 -type d | while read user;
do
   find "$user" -maxdepth 1 -mindepth 1 -type d | while read class;
   do
      mkdir -p "$val_dir/$(basename "$user")/$(basename "$class")"
      find "$class" -maxdepth 1 -mindepth 1 -type f | while read image;
      do
         number=$RANDOM
	 let "number %= $VAL_RANGE"
	 if [ "$number" -lt "1" ]
	 then
	    echo "Moved $image"
	    mv "$image" "$val_dir/$(basename "$user")/$(basename "$class")"
	 fi
      done
   done
done
