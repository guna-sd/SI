# Shell Intelligence (SI) 

```
  ____   _            _  _  ____          _   
 / ___| | |__    ___ | || || __ )   ___  | |_ 
 \___ \ | '_ \  / _ \| || ||  _ \  / _ \ | __|
  ___) || | | ||  __/| || || |_) || (_) || |_ 
 |____/ |_| |_| \___||_||_||____/  \___/  \__|                                               
```

### Shell Intelligence (SI)

Shell Intelligence (SI) is an innovative tool that uses artificial intelligence (AI) to enhance the user experience of the command-line interface (CLI) in Linux. SI enables users to interact with Linux using plain English instead of having to use complex commands.


SI understands natural language input and translates it into commands, making Linux more accessible to a wider audience...

SI provides context-aware error messages and suggestions to help troubleshoot problems effectively and generates reports in plain English, enhancing the understandability of command output.

## Getting Started

To use Shell Intelligence (SI), follow these steps:

##### Install SI on your Linux system.
`git clone https://github.com/guna-sd/SI.git`

run `cd SI`

run `make`

Start using by running `si`

## Example Usage

Here are some examples of how to use SI:

- Set the default text editor to "gedit":
```
guna@anug:[/home/guna/]$!> Set the default text editor to gedit.
```

- Configure the Linux firewall to allow all incoming TCP traffic on port 80:
```
guna@anug:[/home/guna/]$!> Configure the Linux firewall to allow all incoming TCP traffic on port 80.
```

- Generate a report of running processes:
```
guna@anug:[/home/guna/]$!> Generate a report of running processes.
```

## Limitations

While SI offers significant advantages in terms of usability and accessibility, it may have some limitations:

- **Language Understanding**: SI's natural language processing capabilities may not be perfect, leading to occasional misinterpretations of commands.

- **Context Sensitivity**: SI's understanding of context may be limited, resulting in inaccuracies when dealing with complex instructions.

- **Dependency on Language Models**: SI's performance relies on the underlying language model and updates to language models may impact its behavior.

#### development
This project has been tested with pre-trained (base) Models which barely manages. For further improvement and stability the SI will be using Llama3
specifically finetuned for SI with both supervised and Reinforcement Learning methods and quantized model for performance improvements.