#!/usr/bin/awk -f
# Usage: ./checksum.awk ixt-rack-86.log ixt-rack-87.log

BEGIN {
  sum[""]=0
  max_op=0
  max_rank=0
}

{
  if($5=="##") {
    op=strtonum("0x" $7)
    sum[$6 "," op]=$8
    if(op>max_op)
      max_op=op
    if($6>max_rank)
      max_rank=$6
  }
}

END {
  print "number of ops=" max_op+1
  print "number of ranks=" max_rank+1
  for(op=0;op<=max_op;op++) {
    has_op=0
    mismatch=0
    for(rank=0;rank<=max_rank;rank++) {
      val=sum[rank "," op]
      if(rank==0) val_0=val
      if(val!="") {
        has_op++
        if(val!=val_0) mismatch=1
      }
    }
    if(has_op!=0 && (mismatch || has_op<max_rank+1)) {
      printf "%07x: ", op
	    for(rank=0;rank<=max_rank;rank++) {
	      if(sum[rank "," op]!="")
	        printf "%s ", sum[rank "," op]
	      else
	        printf "nan "
	    }
	  printf "\n"
    }
  }
}
