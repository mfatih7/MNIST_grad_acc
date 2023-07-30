import torch.nn as nn

class CNN_Basic(nn.Module):
    def __init__(self):
        super(CNN_Basic, self).__init__()

        #N, C=1, H=28, W=28
##############################################################

        self.cnn1 = nn.Conv2d( in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm1 = nn.BatchNorm2d(16)

        #N, C=16, H=28, W=28        
##############################################################

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm2 = nn.BatchNorm2d(32)        
        
        #N, C=32, H=14, W=14        
##############################################################

        self.cnn3 = nn.Conv2d( in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm3 = nn.BatchNorm2d(64)

        #N, C=64, H=14, W=14        
##############################################################

        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm4 = nn.BatchNorm2d(128)        
        
        #N, C=128, H=7, W=7        
##############################################################

        self.cnn5 = nn.Conv2d( in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm5 = nn.BatchNorm2d(256)

        #N, C=256, H=7, W=7        
##############################################################

        self.cnn6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(2,2), stride=(2,2), padding=(1,1) )           
        self.batchnorm6 = nn.BatchNorm2d(512)        
        
        #N, C=512, H=4, W=4        
##############################################################

        self.cnn7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm7 = nn.BatchNorm2d(1024)        
        
        #N, C=1024, H=2, W=2        
##############################################################

        self.cnn8 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(2,2), stride=(1,1), padding=(0,0) )           
        self.batchnorm8 = nn.BatchNorm2d(2048)        
        
        #N, C=1024, H=1, W=1        
##############################################################  

        self.fc1 = nn.Linear(2048 * 1 * 1, 128)   
        
        #N, C=128, H=1, W=1
##############################################################    

        self.fc2 = nn.Linear(128 * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()
        
    def forward(self, x):
        
        x = self.non_lin( self.batchnorm1( self.cnn1(x) ) )
        x = self.non_lin( self.batchnorm2( self.cnn2(x) ) )
        x = self.non_lin( self.batchnorm3( self.cnn3(x) ) )
        x = self.non_lin( self.batchnorm4( self.cnn4(x) ) )
        x = self.non_lin( self.batchnorm5( self.cnn5(x) ) )
        x = self.non_lin( self.batchnorm6( self.cnn6(x) ) )
        x = self.non_lin( self.batchnorm7( self.cnn7(x) ) )
        x = self.non_lin( self.batchnorm8( self.cnn8(x) ) )
        
        x = x.view( -1, 2048 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

def get_model():
    
    return CNN_Basic( )  


