import torch
import torch.nn as nn

class wnet(nn.Module):
    def __init__(self,input_nc,output_nc,norm_layer=nn.BatchNorm2d):
        super(wnet, self).__init__()

        self.pool = nn.MaxPool2d(2)
        activate_layer = nn.PReLU()

        self.en_down_1_1 = nn.Sequential(nn.Conv2d(input_nc,      output_nc,     3, padding=1), norm_layer(output_nc),     activate_layer)
        self.en_down_2_1 = nn.Sequential(nn.Conv2d(output_nc,     output_nc * 2, 3, padding=1), norm_layer(output_nc * 2), activate_layer)
        self.en_down_3_1 = nn.Sequential(nn.Conv2d(output_nc * 2, output_nc * 4, 3, padding=1), norm_layer(output_nc * 4), activate_layer)

        self.en_up_2_2   = nn.Sequential(nn.ConvTranspose2d(output_nc * 4, output_nc * 2, kernel_size=4, stride=2, padding=1), norm_layer(output_nc * 2), activate_layer)
        self.en_up_1_2   = nn.Sequential(nn.ConvTranspose2d(output_nc * 2, output_nc,     kernel_size=4, stride=2, padding=1), norm_layer(output_nc),     activate_layer)

        self.de_down_1_1 = nn.Sequential(nn.Conv2d(output_nc,     output_nc,     3, padding=1), norm_layer(output_nc),     activate_layer)
        self.de_down_2_1 = nn.Sequential(nn.Conv2d(output_nc,     output_nc * 2, 3, padding=1), norm_layer(output_nc * 2), activate_layer)
        self.de_down_3_1 = nn.Sequential(nn.Conv2d(output_nc * 2, output_nc * 4, 3, padding=1), norm_layer(output_nc * 4), activate_layer)

        self.de_up_2_2   = nn.Sequential(nn.ConvTranspose2d(output_nc * 4, output_nc * 2, kernel_size=4, stride=2, padding=1), norm_layer(output_nc * 2), activate_layer)
        self.de_up_1_2   = nn.Sequential(nn.ConvTranspose2d(output_nc * 2, output_nc,     kernel_size=4, stride=2, padding=1), norm_layer(output_nc), activate_layer)
        self.de_up_1_3   = nn.Sequential(nn.Conv2d(output_nc, input_nc, 3, padding=1), norm_layer(input_nc), activate_layer)

    def forward(self, input):
        en_down_1_1 = self.en_down_1_1(input)

        en_down_2_1 = self.pool(en_down_1_1)
        en_down_2_1 = self.en_down_2_1(en_down_2_1)

        en_down_3_1 = self.pool(en_down_2_1)
        en_down_3_1 = self.en_down_3_1(en_down_3_1)

        en_up_2_2 = self.en_up_2_2(en_down_3_1) + en_down_2_1

        en_up_1_2 = self.en_up_1_2(en_up_2_2) + en_down_1_1

        de_down_1_1 = self.de_down_1_1(en_up_1_2) + en_down_1_1

        de_down_2_1 = self.pool(de_down_1_1)
        de_down_2_1 = self.de_down_2_1(de_down_2_1) + en_down_2_1 + en_up_2_2

        de_down_3_1 = self.pool(de_down_2_1)
        de_down_3_1 = self.de_down_3_1(de_down_3_1) + en_down_3_1

        de_up_2_2 = self.de_up_2_2(de_down_3_1) + de_down_2_1 + en_down_2_1 + en_up_2_2
        de_up_1_2 = self.de_up_1_2(de_up_2_2) + en_down_1_1 + en_up_1_2 + de_down_1_1
        de_up_1_3 = self.de_up_1_3(de_up_1_2)

        return de_up_1_3

class FENNet(nn.Module):
    def __init__(self,input_nc,output_nc,norm_layer=nn.BatchNorm2d):
        super(FENNet, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, padding=0),
            norm_layer(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            norm_layer(64),
            nn.PReLU(),
        )
        self.pool = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.wnet = wnet(64,128)


if __name__ == '__main__':
    model = wnet(64, 128)
    input = torch.ones((1,64,128,128))
    output = model(input)
    # print(model)
    print(input.shape)
    print(output.shape)
