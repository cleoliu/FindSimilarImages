import sys
import win32com.client
import mapPic
 

comNet = win32com.client.Dispatch("WScript.Network")
data = {}
data['remote'] = sys.argv[2] #網路磁碟 #"\\192.168.10.251\share\_cleo\images"
data['local'] = "v:" #本地映射磁碟
 
def ConnectNetworkDrive():      #--連線網路磁碟機--
    comNet.MapNetworkDrive(data['local'], data['remote'])
    localfilepath = data['local']
    return localfilepath

def InterruptNetworkDrive():            #--中斷網路磁碟機--
    comNet.RemoveNetworkDrive(data['local'], True, True)

if __name__ == "__main__":
    ConnectNetworkDrive()
    mapPic.main()
    InterruptNetworkDrive()
