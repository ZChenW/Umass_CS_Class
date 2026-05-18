#!/bin/bash

# Use Unix commands to print the name of each character with dialogue in the script, one name per line.
echo "Question 1.5"
# TODO: Replace the following line with your code.
grep -E "^[[:space:]]{10,}[A-Z][A-Z0-9 .,'-]*([[:space:]]*\\([^)]*\\))?[[:space:]]*$" lotr_script.txt |
	sed -E 's/^ +//; s/ *\([^)]*\)//g; s/ +$//; s/  +/ /g' |
	sort -u
echo "Not Implemented"

# First, extract all lines of dialogue in this script.
# Then, normalize and tokenize this text such that all alphabetic characters are converted to lowercase and words are sequences of alphabetic characers.
# Finally, print the top-20 most frequent word types and their corresponding counts.
# Hint: Ignore parantheticals. These contain short stage directions.
echo
echo "Question 1.6"
# TODO: Replace the following line with your code.
awk '
BEGIN{d=0}
{
  if (match($0,/^[[:space:]]+/) && RLENGTH>=10) {
    name = substr($0, RLENGTH+1)
    if (name ~ /^[A-Z][A-Z0-9 .-]*(\([^)]*\))?[[:space:]]*$/) {
      d=1; next
    }
  }
}
d && /^[[:space:]]*$/ { d=0; next }
d {
  line=$0
  sub(/^[[:space:]]+/, "", line)
  if (line ~ /^\(/) next
  print $0
}
' lotr_script.txt |
	tr 'A-Z' 'a-z' |
	grep -oE '[a-z]+' |
	sort |
	uniq -c |
	sort -nr |
	head -20
echo "Not Implemented"
