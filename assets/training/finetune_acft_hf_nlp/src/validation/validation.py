def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-info", required=True, help="Model source ")

    args = parser.parse_args()

    print("Validation info: ", args.validation_info)
    with open(args.validation_info, "w") as f:
        f.write("Validation Completed")

if __name__ == "__main__":
    main()