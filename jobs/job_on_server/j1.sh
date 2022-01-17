echo "Executing lhc_mask_b1_without_bb_33.json, 0"
which python3
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_33.json -z 0 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_11.json, 0"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_11.json -z 0 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_21.json, 0"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_21.json -z 0 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_33.json, 1"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_33.json -z 1 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_11.json, 1"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_11.json -z 1 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_21.json, 1"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_21.json -z 1 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_33.json, 2"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_33.json -z 2 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_11.json, 2"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_11.json -z 2 -t long 
echo "Done."
echo "Executing lhc_mask_b1_without_bb_21.json, 2"
python3 ../config/generic_run.py --input ../../masks/lhc_mask_b1_without_bb_21.json -z 2 -t long 
echo "Done."
