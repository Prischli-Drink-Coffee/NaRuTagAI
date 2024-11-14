import { Box, Flex, Text, Button, FormControl, Input } from "@chakra-ui/react";


const ContentSection = () => {


  return (
    <Box position="absolute" width="954px" height="380px" left="350px" top="230px">
      {/* Заголовок */}
      <Flex direction="column" align="flex-start" gap="50px">

        <Text
          width="660px"
          height="80px"
          fontFamily="Montserrat"
          fontWeight="700"
          fontSize="32px !important"
          lineHeight="46px"
          color="#023BA3"
        >
            Too lazy to watch the video? Let's say what it's about
        </Text>

        {/* Описание */}
        <Text
          width="600px"
          height="88px"
          fontFamily="Montserrat"
          fontWeight="500"
          fontSize="18px"
          lineHeight="22px"
          color="#1D1D1D"
        >
            NaRuTagAI - allows you to automate the generation of hierarchical tags for videos using multimodal
            artificial intelligence. The system uses visual, audio and text information to make recommendations
            the most suitable tags.
        </Text>

        {/* Поле ввода и кнопка */}
        <Flex direction="row" align="center" gap="0px" width="850px" height="60px">

            {/* Поле ввода URL */}
            <FormControl id="URL" isRequired>
                <Input
                    width="610px"
                    height="60px"
                    border="4px solid #4B8BFC"
                    borderRadius="16px"
                    placeholder="Enter URL here"
                    paddingLeft="20px"
                    bg="#FFFFFF"
                    _placeholder={{
                        fontFamily: "Montserrat",
                        fontWeight: "500",
                        fontSize: "18px",
                        lineHeight: "22px",
                        color: "#1D1D1D"
                    }}
                />
            </FormControl>

          {/* Кнопка */}
          <Button
            width="240px"
            height="60px"
            background="#4B8BFC"
            borderRadius="16px"
            fontFamily="Montserrat"
            fontWeight="700"
            fontSize="18px"
            lineHeight="22px"
            color="#FFFFFF"
            _hover={{ background: "#376fcb" }}
          >
              And what is it about?
          </Button>
        </Flex>
      </Flex>
    </Box>
  );
};

export default ContentSection;
