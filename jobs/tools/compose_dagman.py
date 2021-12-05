import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose a DAGMan file from a list of arguments")
    parser.add_argument("-n", "--number", type=int, help="Number of jobs to run", required=True)
    parser.add_argument("-f", "--first", type=str,
                        help="First job to run", required=True)
    parser.add_argument("-r", "--rest", type=str,
                        help="Rest of the jobs to run", required=True)
    parser.add_argument("-p", "--post", type=str,
                        help="Post script to run", default="")
    
    args = parser.parse_args()

    number = args.number

    # write in the dag file
    with open("dagman.dag", "w") as dagman:
        for i in range(number):
            j = i if args.post == "" else i * 2
            dagman.write(f"JOB JOB{j} {args.first if i == 0 else args.rest}\n")
            if args.post != "":
                dagman.write(f"JOB JOB{j+1} {args.post}\n")

        dagman.write("\n")

        for i in range(number if args.post == "" else number * 2)[:-1]:
            dagman.write(f"PARENT JOB{i} CHILD JOB{i+1}\n")
