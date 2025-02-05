import csv

h = [['0', '0', '0', '0', '0', '0']]

with open(r'C:\data.csv', 'r') as f: 
    reader = csv.reader(f) 
    your_list = list(reader) 
    
   

    
for i in your_list: 
    print(i) 
    if i[-1] == 'yes': 
        j = 0 
        for x in i[:-1]: 
            if x != 'yes': 
                if h[0][j] == '0':  
                    h[0][j] = x 
                elif h[0][j] != x:  
                    h[0][j] = '?'  
            j += 1

print("Maximally Specific set")
print(h)  
     
