export PYTHONPATH=/linux/grotsartdehe/opentps_openhub/opentps_core:$PYTHONPATH

for p in {3..4}
  do
    for i in {0..0}
      do
        python /linux/grotsartdehe/opentps_openhub/waldoScripts/data_generation.py $i $p 1 "train_T2" 21 "FDG" 15
        sleep 5
      done
  done

for p in {6..15}
  do
    for i in {0..0}
      do
        python /linux/grotsartdehe/opentps_openhub/waldoScripts/data_generation.py $i $p 1 "train_T2" 21 "FDG" 15
        sleep 5
      done
  done

for p in {1..13}
  do
    for i in {0..0}
      do
        python /linux/grotsartdehe/opentps_openhub/waldoScripts/data_generation.py $i $p 1 "train_T2" 21 "NO_CPAP" 15
        sleep 5
      done
  done

for p in {15..16}
  do
    for i in {0..0}
      do
        python /linux/grotsartdehe/opentps_openhub/waldoScripts/data_generation.py $i $p 1 "train_T2" 21 "NO_CPAP" 15
        sleep 5
      done
  done

for p in {18..20}
  do
    for i in {0..0}
      do
        python /linux/grotsartdehe/opentps_openhub/waldoScripts/data_generation.py $i $p 1 "train_T2" 21 "NO_CPAP" 15
        sleep 5
      done
  done




