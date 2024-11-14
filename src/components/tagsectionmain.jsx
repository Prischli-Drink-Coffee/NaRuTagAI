import { Box, Flex, Text, Image } from "@chakra-ui/react";
import hashtag from "../images/hashtag.svg";


const TagSection = () => {
    return (
        <Box position="absolute" width="720px" height="159px" left="350px" top="658px">
            {/* Заголовок */}
            <Flex direction="column" align="flex-start" gap="30px">
                <Text
                    width="720px"
                    height="39px"
                    fontFamily="Montserrat"
                    fontWeight="700"
                    fontSize="32px !important"
                    lineHeight="39px"
                    color="#023BA3"
                >
                    As we understand, this is about...
                </Text>

                {/* Теги */}
                <Flex direction="row" align="flex-start" gap="10px" wrap="wrap" width="720px" height="90px">
                    <Box
                        width="87.43px"
                        height="90px"
                        background="transporent"
                        borderRadius="8px"
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                    >

                        <Image src={hashtag} alt="HashTag" />

                    </Box>

                    {/* Массив с тегами */}
                    <Flex direction="row" gap="10px" flexWrap="wrap" width="622.57px" height="90px">
                        {["Sport", "Competition", "Snowboard", "Freestyle", "Girl", "Giraffe Costume"].map((tag, index) => (
                            <Box
                                key={index}
                                width={`${tag.length * 15}px`} // Вычисляем ширину по длине текста
                                height="40px"
                                background="#4B8BFC"
                                borderRadius="12px"
                                display="flex"
                                alignItems="center"
                                justifyContent="center"
                                padding="0px 20px"
                                textAlign="center"
                            >
                                <Text
                                    fontFamily="Montserrat"
                                    fontWeight="700"
                                    fontSize="18px"
                                    lineHeight="22px"
                                    color="#FFFFFF"
                                >
                                    {tag}
                                </Text>
                            </Box>
                        ))}
                    </Flex>
                </Flex>
            </Flex>
        </Box>
    );
};

export default TagSection;
