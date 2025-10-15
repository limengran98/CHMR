if __name__ == "__main__":
    import os
    import datetime

    args = get_args()

    # === Auto-generate model checkpoint name based on hyperparameters ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_tag = (
        f"CHMR"
        f"-lr={args.lr}"
        f"-wdecay={args.wdecay}"
        f"-epoch={args.epochs}"
        f"-batch={args.batch_size}"
        f"-lambda1={args.lambda_1}"
        f"-lambda2={args.lambda_2}"
        f"-decomp={args.decomp_method}"
    )

    # Append user note if provided
    if hasattr(args, "note") and args.note:
        model_tag += f"-{args.note}"

    # Append timestamp for uniqueness
    model_tag += f"-{timestamp}"

    ckpt_dir = "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    args.model_path = os.path.join(ckpt_dir, model_tag + ".pt")

    # === Auto-detect previous checkpoint ===
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("CHMR") and f.endswith(".pt")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
        reverse=True,
    )
    args.resume_ckpt = None

    if ckpt_files:
        latest_ckpt = os.path.join(ckpt_dir, ckpt_files[0])
        print(f"🔍 Detected existing checkpoint: {latest_ckpt}")
        user_choice = input("Do you want to resume training from this checkpoint? (y/n): ").strip().lower()
        if user_choice == "y":
            args.resume_ckpt = latest_ckpt
            print(f"✅ Resuming from checkpoint: {args.resume_ckpt}")
        else:
            print("🚀 Starting a new training run from scratch.")
    else:
        print("🆕 No previous checkpoint found. Starting fresh training.")
        args.resume_ckpt = None

    # === Initialize logger ===
    logger = get_logger(__name__)
    args.logger = logger

    print(f"💾 Model will be saved to: {args.model_path}")
    print(f"📘 Resume checkpoint: {args.resume_ckpt}")
    print(vars(args))

    # === Run main training ===
    main(args, 0)
