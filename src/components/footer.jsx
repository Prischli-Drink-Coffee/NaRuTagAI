import { HStack, Text, Flex } from "@chakra-ui/react";

const Footer = () => {
    return (
        <Flex
            backgroundColor="#4B8BFC"
            width="100%"
            minHeight={["60px", "70px", "80px", "90px", "100px"]}
            padding={["10px", "12px", "15px", "20px", "25px"]}
            justifyContent="center" // Центрирует контент внутри Flex
        >
            <HStack justify="space-between" width="50%">
                <Text
                    color="#FFFFFF"
                    fontFamily="Montserrat"
                    fontSize="18px"
                    lineHeight="22px"
                    fontWeight="500"
                    width="308px"
                    textAlign="right"
                >
                    Команда «Придумать название»
                </Text>
                <HStack
                    onClick={() => {
                        window.scrollTo(0, 0);
                    }}
                >
                    <Text
                        color="#FFFFFF"
                        fontFamily="Montserrat"
                        fontSize="18px"
                        lineHeight="22px"
                        fontWeight="500"
                        width="356px"
                        textAlign="left"
                    >
                        Проект по машинному обучению МТС AI
                    </Text>
                </HStack>
            </HStack>
        </Flex>
    );
};

export default Footer;
