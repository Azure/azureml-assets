from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


if __name__ == "__main__":
    conv_mode = "mpt"
    mm_use_im_start_end = True
    first_message = True

    conv = conv_templates[conv_mode].copy()
    if conv_mode == "mpt":
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    try:
        inp = input(f"{roles[0]}: ")
    except EOFError:
        inp = ""

    if first_message:
        if mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
    else:
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print("input to llava:", prompt)
