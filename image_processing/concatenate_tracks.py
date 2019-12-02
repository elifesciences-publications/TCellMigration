def concatenate_tracks( track_files_list, output_file ):
	import utilities2
	
	master_traj_dict = {}
	for file in track_files_list:
		traj_dict = utilities2.get_trajectory_dict_with_indexes( file )
		
		for traj_ind in traj_dict:
			if traj_ind not in master_traj_dict:
				master_traj_dict[traj_ind] = traj_dict[traj_ind]
			else:
				for entry in traj_dict[traj_ind]:
					master_traj_dict[traj_ind].append( entry )
	
	tracks_file = open(output_file, 'w')
	
	for ind in master_traj_dict:
		track = master_traj_dict[ind]
		output_strs = [str(ind)]
		for entry in track:
		
			output_list1 = list(entry)
			output_list1[0] = int(entry[0])
			output_list = [str(s) for s in output_list1]
			output_strs.append( (',').join(output_list) )
		tracks_file.write(('\t').join( output_strs ) + '\n')
	
	tracks_file.close()

if __name__ == "__main__":
	
	import sys
	
	track_files_list = sys.argv[1:-1]
	output_file = sys.argv[-1]
	
	concatenate_tracks( track_files_list, output_file )
	