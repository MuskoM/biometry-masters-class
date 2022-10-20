import argparse
import sys
import ps1.main as ps1

if __name__ == '__main__':
    lesson = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--ps')
    parser.add_argument('-z', '--zad')
    args = parser.parse_args()

    lesson = args.ps
    exercise = args.zad

    lessons = {
        '1' : ps1.run
    }
    
    print(f'Starting PS {lesson}')
    lessons[lesson](exercise)